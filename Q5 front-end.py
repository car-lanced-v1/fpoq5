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
        self.var_map = {}
        self.var_names = [] 

    def _get_var_index(self, var_name):
        if var_name not in self.var_map:
            self.var_map[var_name] = len(self.var_names)
            self.var_names.append(var_name)
        return self.var_map[var_name]

    def _parse_expression(self, expr):
        """Lê uma expressão linear (ex: '2x1 - 3x2') e retorna coeficientes."""
        expr = expr.replace(" ", "")
        # Regex para capturar termo: (sinal?)(numero?)(variavel)
        pattern = r'([+-]?\d*\.?\d*)([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, expr)
        
        coeffs = {}
        for coeff_str, var_name in matches:
            idx = self._get_var_index(var_name)
            if coeff_str in ['', '+']: val = 1.0
            elif coeff_str == '-': val = -1.0
            else: val = float(coeff_str)
            coeffs[idx] = coeffs.get(idx, 0) + val
        return coeffs

    def parse(self, text):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines: raise ValueError("O texto de entrada está vazio.")

        sense = 'max'
        c_coeffs = {}
        A_ub, b_ub = [], []
        A_eq, b_eq = [], []
        bounds_map = {} # Para casos específicos como x1 >= 4

        # 1. Identificar Objetivo
        obj_line = lines[0].lower()
        if 'minimizar' in obj_line or 'min' in obj_line: sense = 'min'
        elif 'maximizar' in obj_line or 'max' in obj_line: sense = 'max'
        else: raise ValueError("A primeira linha deve indicar 'Maximizar' ou 'Minimizar'.")

        if '=' not in lines[0]:
             raise ValueError("A função objetivo deve conter '=' (Ex: Maximizar Z = ...)")
        
        obj_expr = lines[0].split('=')[1]
        c_coeffs = self._parse_expression(obj_expr)

        # 2. Processar Restrições
        constraints_started = False
        for line in lines[1:]:
            line_lower = line.lower()
            if 'sujeito a' in line_lower or 'restrições' in line_lower:
                constraints_started = True
                continue
            
            # Pula comentários ou restrições de sinal genéricas (x >= 0)
            if not constraints_started or line.startswith('#'): continue
            if '>= 0' in line and ',' in line: continue # Ignora "x1, x2 >= 0"

            # Detectar operador
            operator = None
            if '<=' in line: operator = '<='
            elif '>=' in line: operator = '>='
            elif '=' in line: operator = '='
            
            if not operator: continue

            lhs, rhs = line.split(operator)
            try: rhs_val = float(rhs.strip())
            except: continue # Pula se não conseguir ler o número direito

            row_coeffs = self._parse_expression(lhs)

            # Normalização
            if operator == '<=':
                A_ub.append((row_coeffs, rhs_val))
            elif operator == '>=':
                # Se for apenas uma variável (ex: x2 >= 4), é um bound
                if len(row_coeffs) == 1 and list(row_coeffs.values())[0] == 1.0:
                    var_idx = list(row_coeffs.keys())[0]
                    current_min, current_max = bounds_map.get(var_idx, (0, None))
                    bounds_map[var_idx] = (max(current_min, rhs_val), current_max)
                else:
                    # Restrição normal invertida
                    neg_coeffs = {k: -v for k, v in row_coeffs.items()}
                    A_ub.append((neg_coeffs, -rhs_val))
            elif operator == '=':
                A_eq.append((row_coeffs, rhs_val))

        # 3. Montar Matrizes
        n_vars = len(self.var_names)
        c = np.zeros(n_vars)
        for idx, val in c_coeffs.items(): c[idx] = val

        mat_A_ub = np.zeros((len(A_ub), n_vars)) if A_ub else None
        vec_b_ub = np.zeros(len(A_ub)) if A_ub else None
        if A_ub:
            for i, (coeffs, val) in enumerate(A_ub):
                vec_b_ub[i] = val
                for idx, v in coeffs.items(): mat_A_ub[i, idx] = v

        mat_A_eq = np.zeros((len(A_eq), n_vars)) if A_eq else None
        vec_b_eq = np.zeros(len(A_eq)) if A_eq else None
        if A_eq:
            for i, (coeffs, val) in enumerate(A_eq):
                vec_b_eq[i] = val
                for idx, v in coeffs.items(): mat_A_eq[i, idx] = v
        
        # Consolida bounds
        final_bounds = []
        for i in range(n_vars):
            final_bounds.append(bounds_map.get(i, (0, None)))

        return {
            'c': c, 'A_ub': mat_A_ub, 'b_ub': vec_b_ub,
            'A_eq': mat_A_eq, 'b_eq': vec_b_eq,
            'sense': sense, 'var_names': self.var_names,
            'bounds': final_bounds
        }

# ==============================================================================
# 2. MOTOR LÓGICO: BRANCH AND BOUND SOLVER
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
        self.base_bounds = bounds if bounds else [(0, None)] * len(c)
        
        self.best_int_solution = None
        self.best_int_value = -np.inf if sense == 'max' else np.inf
        self.tree_log = [] 
        self.nodes_count = 0

    def solve_relaxed(self, current_bounds):
        return linprog(c=self.c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq,
                       bounds=current_bounds, method='highs')

    def is_integer(self, x):
        return np.all(np.abs(x - np.round(x)) < 1e-5)

    def get_branching_variable(self, x):
        frac_parts = np.abs(x - np.round(x))
        candidates = np.where(frac_parts > 1e-5)[0]
        if len(candidates) == 0: return None, None
        return candidates[np.argmax(frac_parts[candidates])], x[candidates[np.argmax(frac_parts[candidates])]]

    def _fmt(self, v):
        if v is None: return "-"
        if isinstance(v, (list, np.ndarray)): return str(tuple([int(round(i)) for i in v])).replace("'", "")
        if abs(v - round(v)) < 1e-5: return f"{int(round(v))}"
        return f"{v:.4f}"

    def solve(self):
        queue = [{'bounds': copy.deepcopy(self.base_bounds), 'id': 1, 'parent': 0, 'constraint': 'Raiz'}]
        self.nodes_count = 0
        
        while queue:
            node = queue.pop(0)
            self.nodes_count += 1
            res = self.solve_relaxed(node['bounds'])
            z_val = -res.fun if self.sense == 'max' and res.success else res.fun
            
            node_rec = {'id': node['id'], 'parent': node['parent'], 'constraint': node['constraint'], 
                        'z': z_val, 'x': res.x if res.success else None, 'status': '', 'pruned': False}

            if not res.success:
                node_rec.update({'status': "Infactível", 'pruned': True})
                self.tree_log.append(node_rec); continue

            is_pruned = False
            if self.best_int_solution is not None:
                if (self.sense == 'max' and z_val <= self.best_int_value + 1e-6) or \
                   (self.sense == 'min' and z_val >= self.best_int_value - 1e-6): is_pruned = True
            
            if is_pruned:
                node_rec.update({'status': f"Poda (Z={z_val:.2f})", 'pruned': True})
                self.tree_log.append(node_rec); continue

            if self.is_integer(res.x):
                node_rec.update({'status': "Inteira", 'pruned': True})
                update = False
                if self.best_int_solution is None: update = True
                elif self.sense == 'max' and z_val > self.best_int_value: update = True
                elif self.sense == 'min' and z_val < self.best_int_value: update = True
                
                if update:
                    self.best_int_value = z_val
                    self.best_int_solution = res.x
                    node_rec['status'] += " (Nova Melhor!)"
                else: node_rec['status'] += " (Sub-ótima)"
                self.tree_log.append(node_rec); continue

            idx, val = self.get_branching_variable(res.x)
            node_rec['status'] = f"Ramificar {self.var_names[idx]}"
            self.tree_log.append(node_rec)
            
            floor_v, ceil_v = np.floor(val), np.ceil(val)
            lb, rb = copy.deepcopy(node['bounds']), copy.deepcopy(node['bounds'])
            
            lb[idx] = (lb[idx][0], min(lb[idx][1] if lb[idx][1] is not None else np.inf, floor_v))
            rb[idx] = (max(rb[idx][0], ceil_v), rb[idx][1])

            next_id = self.nodes_count + len(queue) + 1
            queue.append({'bounds': lb, 'id': next_id, 'parent': node['id'], 'constraint': f"{self.var_names[idx]} <= {floor_v:.0f}"})
            queue.append({'bounds': rb, 'id': next_id+1, 'parent': node['id'], 'constraint': f"{self.var_names[idx]} >= {ceil_v:.0f}"})

        return self.best_int_solution, self.best_int_value

# ==============================================================================
# 3. INTERFACE STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(page_title="Solver B&B - FPO", page_icon="🌳", layout="wide")

    st.title("🌳 Solver de Programação Inteira (Branch and Bound)")
    st.markdown("""
    **Trabalho Final de Pesquisa Operacional - Questão 5(c)**
    
    Esta ferramenta permite resolver modelos customizados ou carregar os exercícios da lista oficial.
    O algoritmo exibe a árvore de decisão completa e permite validação dos resultados.
    """)
    
    st.divider()

    # --- Dicionário de Exercícios Completo ---
    # A notação >= 0 é adicionada ao final para clareza visual, embora o parser assuma default.
    exercises = {
        "Selecione um exercício...": "",
        "Ex 17: Energéticos": "Minimizar Z = 0.06x1 + 0.08x2\nSujeito a:\n8x1 + 6x2 >= 48\n1x1 + 2x2 >= 12\n1x1 + 2x2 <= 20\nx1, x2 >= 0",
        "Ex 18: Quinquilharias": "Maximizar Z = 2x1 + 1x2\nSujeito a:\n6x1 + 3x2 <= 480\n2x1 + 4x2 <= 480\nx1, x2 >= 0",
        "Ex 21: Janelas": "Maximizar Z = 60x1 + 30x2\nSujeito a:\nx1 <= 6\nx2 <= 4\n6x1 + 8x2 <= 48\nx1, x2 >= 0",
        "Ex 26a: Forma Padrão": "Maximizar Z = 2x1 - 1x2 + 1x3\nSujeito a:\n3x1 + x2 + x3 <= 60\nx1 - x2 + x3 <= 10\nx1 + x2 - x3 <= 20\nx1, x2, x3 >= 0",
        "Ex 26b: Forma Padrão (Igualdade)": "Minimizar Z = -2x1 + 0.75x2 - 12x3\nSujeito a:\nx1 - 5x3 <= 3\nx1 + x2 = 12\n-3x1 - x2 + x3 <= 27\nx1, x2, x3 >= 0",
        "Ex 26c: Variável Livre (u-v)": "Maximizar Z = 4u - 4v + 3x2\nSujeito a:\nu - v + 3x2 <= 7\n2u - 2v + 2x2 <= 8\nu - v + x2 <= 3\nx2 <= 2\nu, v, x2 >= 0",
        "Ex 26d: Bounds Específicos": "Minimizar Z = 8x1 + 10x2\nSujeito a:\n-x1 + x2 <= 2\n-4x1 - 5x2 <= -20\nx1 <= 6\nx2 >= 4",
        "Ex 26e: Misto": "Minimizar Z = 2x1 + 6x2\nSujeito a:\n-4x1 - 3x2 <= -12\n2x1 + x2 = 8\nx1, x2 >= 0",
        "Ex 28: Malas e Mochilas": "Maximizar Z = 50x1 + 40x2\nSujeito a:\n2x1 <= 300\n3x2 <= 540\n2x1 + 2x2 <= 440\n1.2x1 + 1.5x2 <= 300\nx1, x2 >= 0",
        "Ex 29: Canhões": "Maximizar Z = 23x1 + 32x2\nSujeito a:\n10x1 + 6x2 <= 2500\n5x1 + 10x2 <= 2000\n1x1 + 2x2 <= 500\nx1, x2 >= 0",
        "Ex 31: Bens Capital": "Maximizar Z = 3x1 - 2x2 + 6x3\nSujeito a:\nx1 + x2 + 2x3 = 12\n2x1 + 3x2 + 12x3 <= 48\nx1, x2, x3 >= 0"
    }

    # --- Área de Seleção e Edição (Layout Melhorado) ---
    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.subheader("1. Entrada de Dados")
        
        # Selectbox acima da área de texto
        selected_key = st.selectbox("📚 Carregar Exercício da Lista:", list(exercises.keys()))
        
        # Área de texto populada dinamicamente
        input_value = exercises[selected_key]
        problem_text = st.text_area("Edite o modelo matemático aqui:", value=input_value, height=350,
                                    placeholder="Maximizar Z = ...\nSujeito a:\n...")
        
        solve_btn = st.button("🚀 Resolver Modelo", type="primary", use_container_width=True)

    with col_info:
        st.info("""
        **Guia de Sintaxe:**
        1. **Objetivo:** Comece com 'Maximizar' ou 'Minimizar'. Use '='.
        2. **Restrições:** Use `Sujeito a:` seguido das equações.
        3. **Operadores:** Use `<=`, `>=` ou `=`.
        4. **Variáveis:** Use nomes como `x1`, `xA`, `y`.
        5. **Não-negatividade:** O sistema assume `x >= 0` por padrão, mas você pode escrever para clareza.
        
        *Exemplo:*
        ```text
        Maximizar Z = 3x1 + 5x2
        Sujeito a:
        x1 <= 4
        2x2 <= 12
        3x1 + 2x2 = 18
        x1, x2 >= 0
        ```
        """)

    st.divider()

    # --- Processamento ---
    if solve_btn and problem_text:
        try:
            # 1. Parsing
            parser = ModelParser()
            model_data = parser.parse(problem_text)
            
            # 2. Feedback Matemático (LaTeX)
            st.subheader("2. Interpretação do Modelo")
            
            # Formatação da FO
            sense = model_data['sense'].title()
            terms = []
            for i, c in enumerate(model_data['c']):
                if abs(c) > 1e-5:
                    s = f"+ {c:.2g}" if c > 0 else f"- {abs(c):.2g}"
                    if i == 0 and c > 0: s = f"{c:.2g}"
                    terms.append(f"{s}{model_data['var_names'][i]}")
            st.latex(f"\\text{{{sense}}} \\quad Z = {' '.join(terms)}")
            
            # Formatação das Restrições
            st.markdown("**Restrições Identificadas:**")
            
            # UB
            if model_data['A_ub'] is not None:
                for i, row in enumerate(model_data['A_ub']):
                    lhs = " + ".join([f"{v:.2g}{model_data['var_names'][j]}" for j, v in enumerate(row) if abs(v)>1e-5])
                    st.latex(f"{lhs} \\le {model_data['b_ub'][i]:.2g}")
            
            # EQ
            if model_data['A_eq'] is not None:
                for i, row in enumerate(model_data['A_eq']):
                    lhs = " + ".join([f"{v:.2g}{model_data['var_names'][j]}" for j, v in enumerate(row) if abs(v)>1e-5])
                    st.latex(f"{lhs} = {model_data['b_eq'][i]:.2g}")
            
            st.latex(f"{', '.join(model_data['var_names'])} \\ge 0, \\in \\mathbb{{Z}}")

            # 3. Solver
            solver = BranchAndBoundSolver(
                c=model_data['c'], A_ub=model_data['A_ub'], b_ub=model_data['b_ub'],
                A_eq=model_data['A_eq'], b_eq=model_data['b_eq'], bounds=model_data['bounds'],
                sense=model_data['sense'], var_names=model_data['var_names']
            )
            
            with st.spinner("Calculando solução ótima inteira..."):
                best_sol, best_val = solver.solve()

            # 4. Resultados
            st.subheader("3. Resultados")
            col_res1, col_res2 = st.columns(2)
            
            if best_sol is not None:
                with col_res1:
                    st.success("✅ Solução Ótima Inteira Encontrada")
                    st.metric("Função Objetivo (Z*)", f"{best_val:.4f}")
                with col_res2:
                    st.write("**Variáveis de Decisão (x*):**")
                    # Formatação de Tupla (x1, x2, ...)
                    tuple_str = str(tuple([int(round(v)) for v in best_sol])).replace("'", "")
                    st.code(tuple_str, language="text")
            else:
                st.error("⚠️ Não foi encontrada solução inteira viável para as restrições dadas.")

            # 5. Tabela de Auditoria (Log)
            st.subheader("4. Detalhes da Execução (Árvore de Decisão)")
            
            tree_data = []
            for node in solver.tree_log:
                tree_data.append({
                    "ID": node['id'],
                    "Pai": node['parent'],
                    "Restrição Adicionada": node['constraint'],
                    "Z (Relaxado)": solver._fmt(node['z']) if node['z'] else "-",
                    "Status / Ação": node['status'],
                    "Solução Parcial (x)": solver._fmt(node['x']) if node['x'] is not None else "-"
                })
            
            st.dataframe(pd.DataFrame(tree_data), use_container_width=True, hide_index=True)

            # 6. Relatório para Download
            report_lines = [
                "RELATÓRIO FINAL - SOLVER BRANCH AND BOUND",
                "="*50,
                f"MODELO ORIGINAL:\n{problem_text}",
                "-"*50,
                "RESULTADO FINAL:",
                f"Status: {'Sucesso' if best_sol is not None else 'Inviável'}",
                f"Z*: {best_val:.4f}",
                f"x*: {tuple([int(round(v)) for v in best_sol]) if best_sol is not None else '-'}",
                "-"*50,
                "HISTÓRICO DA ÁRVORE:",
                pd.DataFrame(tree_data).to_string(index=False)
            ]
            
            st.download_button(
                label="📥 Baixar Relatório (.txt)",
                data="\n".join(report_lines),
                file_name="resultado_branch_bound.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Erro ao processar o modelo: {str(e)}")
            st.warning("Verifique a sintaxe. Certifique-se de usar pontos para decimais (ex: 1.5) e não vírgulas.")

if __name__ == "__main__":
    main()
