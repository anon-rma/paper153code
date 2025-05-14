import pandas as pd
import ast

def compute_multi_tsv_inner_file(file_list, selected_plans):    
    acc_dfs = []

    for path in file_list:
        df = pd.read_csv(path, sep='\t', dtype=str)
        df['gt_plan'] = df['gt'].apply(
            lambda x: ast.literal_eval(x).get('plan') if pd.notnull(x) else None
        )
        df['correct'] = df['all'].str.lower() == 'pass'
        df_sel = df[df['gt_plan'].isin(selected_plans)]
        df_sel = df
        if df_sel.empty:            
            continue

        pivot = df_sel.pivot_table(
            index='gt_plan',
            columns='file',
            values='correct',
            aggfunc='mean'
        ) * 100

        new_cols = {col: f"{path}|{col}" for col in pivot.columns}
        pivot = pivot.rename(columns=new_cols)
        acc_dfs.append(pivot)

    if not acc_dfs:        
        return

    result = pd.concat(acc_dfs, axis=1)
    sorted_cols = sorted(
        result.columns,
        key=lambda col: col.split('|', 1)[1]
    )
    result = result[sorted_cols]

    macro_avg = result.mean(axis=0, skipna=True)
    result.loc['Macro'] = macro_avg
    print(result.fillna("N/A").to_string(float_format="{:.4f}".format))

    result.to_csv(
        "evaluation_result.tsv",
        sep='\t',
        index=True,
        na_rep="N/A",
        float_format="%.2f"
    )


if __name__ == "__main__":
    file_list = [        
        "phi4-rma-addi-complex.tsv",
        "phi4-history-complex.tsv",        
    ]
    selected_plans = [
        'ACTION_EDIT_ALARM', 'ACTION_EDIT_CONTACT', 'ACTION_EDIT_DOCUMENT',
        'ACTION_EDIT_VIDEO', 'ACTION_INSERT_CONTACT', 'ACTION_INSERT_EVENT',
        'ACTION_NAVIGATE_TO_LOCATION', 'ACTION_OPEN_CONTENT',
        'dial', 'play_music', 'play_video', 'search_location',
        'send_email', 'send_message'
    ]
    compute_multi_tsv_inner_file(file_list, selected_plans)
