import streamlit as st
import numpy as np
from scipy.optimize import linprog
import re
import copy
import pandas as pd

# ==============================================================================
# 1. MOTOR LÓGICO: PARSER DE LINGUAGEM NATURAL
# ==============================================================================
class ModelParser:
    def __init__(self):
        self.var_map = {}  # Mapeia 'x1' -> 0, 'x2' -> 1
        self.var_names = [] # Lista ordenada de nomes

    def _get_var_index(self, var_name):
        if var_name not in self.var_map:
            self.var_map[var_name] = len(self.var_names)
            self.var_names.append(var_name)
        return self.var_map[var_name]

    def _parse_expression(self, expr):
        """Lê uma expressão linear (ex: '2x1 - 3x2') e retorna coeficientes."""
        # Remove espaços e padroniza
        expr = expr.replace(" ", "")
        
        # Regex para capturar termo: (sinal?)(numero?)(variavel)
        # Ex: -2x1, +x2, x3, -x4
        pattern = r'([+-]?\d*\.?\d*)([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, expr)
        
        coeffs = {}
        for coeff_str, var_name in matches:
            idx = self._get_var_index(var_name)
            
            # Tratar coeficientes implícitos
            if coeff_str in ['', '+']: val = 1.0
            elif coeff_str == '-': val = -1.0
            else: val = float(coeff_str)
            
            coeffs[idx] = coeffs.get(idx, 0) + val
            
        return coeffs

    def parse(self, text):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            raise ValueError("O texto de entrada está vazio.")

        sense = 'max'
        c_coeffs = {}
        A_ub, b_ub = [], []
        A_eq, b_eq = [], []

        # 1. Identificar Objetivo
        obj_line = lines[0].lower()
        if 'minimizar' in obj_line or 'min' in obj_line:
            sense = 'min'
        elif 'maximizar' in obj_line or 'max' in obj_line:
            sense = 'max'
        else:
            raise ValueError("A primeira linha deve indicar 'Maximizar' ou 'Minimizar'.")

        # Parse Função Objetivo (Lado direito do '=')
        if '=' not in lines[0]:
             raise ValueError("A função objetivo deve conter '=' (Ex: Maximizar Z = 2x1 + 3x2)")
        
        obj_expr = lines[0].split('=')[1]
        c_coeffs = self._parse_expression(obj_expr)

        # 2. Processar Restrições
        constraints_started = False
        for line in lines[1:]:
            line_lower = line.lower()
            if 'sujeito a' in line_lower or 'restrições' in line_lower:
                constraints_started = True
                continue
            
            # Ignorar linhas de comentário ou vazias
            if not constraints_started or line.startswith('#'):
                continue

            # Detectar operador
            operator = None
            if '<=' in line: operator = '<='
            elif '>=' in line: operator = '>='
            elif '=' in line: operator = '=' # Igualdade estrita
            
            if not operator:
                # Pode ser linha de declaração de variáveis (ignorar por enquanto, assume >= 0)
                continue

            lhs, rhs = line.split(operator)
            try:
                rhs_val = float(rhs.strip())
            except:
                raise ValueError(f"O lado direito da restrição deve ser um número: '{line}'")

            row_coeffs = self._parse_expression(lhs)

            # Normalização para Solver (Ax <= b ou Ax = b)
            if operator == '<=':
                # Mantém
                A_ub.append((row_coeffs, rhs_val))
            elif operator == '>=':
                # Multiplica por -1 para virar <=
                neg_coeffs = {k: -v for k, v in row_coeffs.items()}
                A_ub.append((neg_coeffs, -rhs_val))
            elif operator == '=':
                A_eq.append((row_coeffs, rhs_val))

        # 3. Montar Matrizes Finais
        n_vars = len(self.var_names)
        
        # Vetor C
        c = np.zeros(n_vars)
        for idx, val in c_coeffs.items():
            c[idx] = val

        # Matrizes UB
        mat_A_ub = np.zeros((len(A_ub), n_vars)) if A_ub else None
        vec_b_ub = np.zeros(len(A_ub)) if A_ub else None
        
        if A_ub:
            for i, (coeffs, val) in enumerate(A_ub):
                vec_b_ub[i] = val
                for idx, v in coeffs.items():
                    mat_A_ub[i, idx] = v

        # Matrizes EQ
        mat_A_eq = np.zeros((len(A_eq), n_vars)) if A_eq else None
        vec_b_eq = np.zeros(len(A_eq)) if A_eq else None
        
        if A_eq:
            for i, (coeffs, val) in enumerate(A_eq):
                vec_b_eq[i] = val
                for idx, v in coeffs.items():
                    mat_A_eq[i, idx] = v

        return {
            'c': c,
            'A_ub': mat_A_ub, 'b_ub': vec_b_ub,
            'A_eq': mat_A_eq, 'b_eq': vec_b_eq,
            'sense': sense,
            'var_names': self.var_names
        }

# ==============================================================================
# 2. MOTOR LÓGICO: BRANCH AND BOUND SOLVER (O Mesmo do Notebook)
# ==============================================================================
class BranchAndBoundSolver:
    def __init__(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense='max', var_names=None):
        self.sense = sense
        self.c_original = np.array(c)
        self.c = -np.array(c) if sense == 'max' else np.array(c)
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.var_names = var_names if var_names else [f"x{i+1}" for i in range(len(c))]
        
        n_vars = len(c)
        self.base_bounds = [(0, None)] * n_vars
        
        self.best_int_solution = None
        self.best_int_value = -np.inf if sense == 'max' else np.inf
        self.tree_log = [] 
        self.nodes_count = 0

    def solve_relaxed(self, current_bounds):
        return linprog(
            c=self.c, A_ub=self.A_ub, b_ub=self.b_ub,
            A_eq=self.A_eq, b_eq=self.b_eq,
            bounds=current_bounds, method='highs'
        )

    def is_integer(self, x):
        return np.all(np.abs(x - np.round(x)) < 1e-5)

    def get_branching_variable(self, x):
        fractional_parts = np.abs(x - np.round(x))
        candidates = np.where(fractional_parts > 1e-5)[0]
        if len(candidates) == 0: return None, None
        best_idx = candidates[np.argmax(fractional_parts[candidates])]
        return best_idx, x[best_idx]

    def _format_value(self, v):
        if abs(v - round(v)) < 1e-5: return f"{int(round(v))}"
        return f"{v:.4f}"

    def solve(self):
        n_vars = len(self.c)
        queue = [{'bounds': copy.deepcopy(self.base_bounds), 'id': 1, 'parent': 0, 'constraint': 'Raiz'}]
        self.nodes_count = 0
        
        while queue:
            node = queue.pop(0)
            self.nodes_count += 1
            
            res = self.solve_relaxed(node['bounds'])
            z_val = -res.fun if self.sense == 'max' and res.success else res.fun
            
            node_record = {
                'id': node['id'], 'parent': node['parent'], 
                'constraint': node['constraint'], 'z': z_val, 
                'x': res.x if res.success else None,
                'status': '', 'pruned': False
            }

            if not res.success:
                node_record['status'] = "Infactível"
                node_record['pruned'] = True
                self.tree_log.append(node_record)
                continue

            is_bound_pruned = False
            if self.best_int_solution is not None:
                if self.sense == 'max' and z_val <= self.best_int_value + 1e-6: is_bound_pruned = True
                elif self.sense == 'min' and z_val >= self.best_int_value - 1e-6: is_bound_pruned = True
            
            if is_bound_pruned:
                node_record['status'] = f"Poda (Z={z_val:.2f})"
                node_record['pruned'] = True
                self.tree_log.append(node_record)
                continue

            if self.is_integer(res.x):
                node_record['status'] = "Inteira"
                node_record['pruned'] = True
                update = False
                if self.best_int_solution is None: update = True
                elif self.sense == 'max' and z_val > self.best_int_value: update = True
                elif self.sense == 'min' and z_val < self.best_int_value: update = True
                
                if update:
                    self.best_int_value = z_val
                    self.best_int_solution = res.x
                    node_record['status'] += " (Nova Melhor!)"
                else:
                    node_record['status'] += " (Sub-ótima)"
                self.tree_log.append(node_record)
                continue

            idx, val = self.get_branching_variable(res.x)
            node_record['status'] = f"Ramificar {self.var_names[idx]}"
            self.tree_log.append(node_record)
            
            floor_v, ceil_v = np.floor(val), np.ceil(val)
            
            left_bounds = copy.deepcopy(node['bounds'])
            l_min, l_max = left_bounds[idx]
            left_bounds[idx] = (l_min, floor_v if l_max is None else min(l_max, floor_v))
            
            right_bounds = copy.deepcopy(node['bounds'])
            r_min, r_max = right_bounds[idx]
            right_bounds[idx] = (max(r_min, ceil_v), r_max)

            next_id = self.nodes_count + len(queue) + 1
            queue.append({'bounds': left_bounds, 'id': next_id, 'parent': node['id'], 'constraint': f"{self.var_names[idx]} <= {floor_v:.0f}"})
            queue.append({'bounds': right_bounds, 'id': next_id+1, 'parent': node['id'], 'constraint': f"{self.var_names[idx]} >= {ceil_v:.0f}"})

        return self.best_int_solution, self.best_int_value

# ==============================================================================
# 3. INTERFACE STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(page_title="Solver Branch & Bound", page_icon="🌳", layout="wide")

    st.title("🌳 Solver de Programação Inteira (Branch and Bound)")
    st.markdown("""
    Esta ferramenta resolve problemas de **Programação Linear Inteira (PLI)** utilizando o algoritmo **Branch and Bound**.
    Ela foi desenvolvida para atender aos critérios da **Questão 5(c)** do Trabalho Final de FPO.
    
    **Características:**
    * Entrada em notação matemática natural (humana).
    * Visualização da árvore de decisão e critérios de corte.
    * Relatório completo para download.
    """)

    # --- Sidebar: Exemplos ---
    st.sidebar.header("Carregar Exemplo")
    example_options = {
        "Limpar": "",
        "Ex 17: Energéticos (Min)": "Minimizar Z = 0.06x1 + 0.08x2\nSujeito a:\n8x1 + 6x2 >= 48\n1x1 + 2x2 >= 12\n1x1 + 2x2 <= 20",
        "Ex 18: Quinquilharias (Max)": "Maximizar Z = 2x1 + 1x2\nSujeito a:\n6x1 + 3x2 <= 480\n2x1 + 4x2 <= 480",
        "Ex 31: Bens Capital (Igualdade)": "Maximizar Z = 3x1 - 2x2 + 6x3\nSujeito a:\nx1 + x2 + 2x3 = 12\n2x1 + 3x2 + 12x3 <= 48"
    }
    
    selected_example = st.sidebar.selectbox("Escolha um exercício da lista:", list(example_options.keys()))
    
    # --- Área de Entrada ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Definição do Modelo")
        st.markdown("**Formato Aceito:**")
        st.code("""Maximizar Z = 2x1 + 3x2
Sujeito a:
x1 + x2 <= 10
2x1 - x2 >= 5""")
        
        default_text = example_options[selected_example]
        problem_text = st.text_area("Digite ou cole seu modelo abaixo:", value=default_text, height=300)
        
        solve_btn = st.button("🚀 Resolver Modelo", type="primary")

    # --- Processamento e Resultados ---
    if solve_btn and problem_text:
        with col2:
            try:
                # 1. Parsing
                parser = ModelParser()
                model_data = parser.parse(problem_text)
                
                # 2. Mostra interpretação matemática (Feedback visual)
                st.subheader("2. Interpretação do Modelo")
                
                # Renderiza função objetivo em LaTeX
                sense_str = model_data['sense'].title()
                obj_terms = []
                for i, c in enumerate(model_data['c']):
                    coeff = f"{c:.2f}".rstrip('0').rstrip('.')
                    if c >= 0 and i > 0: coeff = f"+ {coeff}"
                    obj_terms.append(f"{coeff}{model_data['var_names'][i]}")
                st.latex(f"\\text{{{sense_str}}} \\quad Z = {' '.join(obj_terms)}")
                
                # 3. Resolução
                solver = BranchAndBoundSolver(
                    c=model_data['c'],
                    A_ub=model_data['A_ub'], b_ub=model_data['b_ub'],
                    A_eq=model_data['A_eq'], b_eq=model_data['b_eq'],
                    sense=model_data['sense'],
                    var_names=model_data['var_names']
                )
                
                with st.spinner("Executando Branch and Bound..."):
                    best_sol, best_val = solver.solve()

                # 4. Resultados Finais
                st.subheader("3. Resultado Final")
                
                if best_sol is not None:
                    st.success(f"**Solução Ótima Inteira Encontrada!**")
                    st.metric(label="Valor da Função Objetivo (Z*)", value=f"{best_val:.4f}")
                    
                    # Tabela de Variáveis
                    sol_dict = {name: int(round(val)) for name, val in zip(model_data['var_names'], best_sol)}
                    st.write("**Variáveis de Decisão:**")
                    st.json(sol_dict)
                else:
                    st.error("O problema não possui solução inteira viável.")

                # 5. Relatório Detalhado (Log)
                report_text = f"RELATÓRIO DE RESOLUÇÃO\n{'='*40}\n\n"
                report_text += f"MODELO DE ENTRADA:\n{problem_text}\n\n"
                report_text += f"{'-'*40}\nÁRVORE DE DECISÃO (Branch & Bound)\n{'-'*40}\n"
                report_text += f"{'ID':<4} | {'Pai':<4} | {'Restrição':<15} | {'Z Relaxado':<12} | {'Status'}\n"
                
                tree_data = []
                for node in solver.tree_log:
                    z_str = f"{node['z']:.4f}" if node['z'] is not None else "---"
                    report_text += f"{node['id']:<4} | {node['parent']:<4} | {node['constraint']:<15} | {z_str:<12} | {node['status']}\n"
                    tree_data.append({
                        "ID": node['id'], "Pai": node['parent'], 
                        "Restrição": node['constraint'], "Z": z_str, 
                        "Status": node['status'],
                        "Solução (X)": str([solver._format_value(v) for v in node['x']]) if node['x'] is not None else ""
                    })
                
                st.subheader("4. Árvore de Decisão Detalhada")
                st.dataframe(pd.DataFrame(tree_data), use_container_width=True)

                # 6. Botão de Download
                st.download_button(
                    label="📥 Baixar Relatório Completo (.txt)",
                    data=report_text,
                    file_name="solucao_branch_bound.txt",
                    mime="text/plain"
                )

            except ValueError as ve:
                st.error(f"❌ Erro de Sintaxe: {ve}")
                st.info("Verifique se usou a notação correta (ex: '2x1 + 3x2 <= 10').")
            except Exception as e:
                st.error(f"❌ Erro Inesperado: {e}")

if __name__ == "__main__":
    main()