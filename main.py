import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from datetime import timedelta
from numpy.linalg import norm


def calculate_similarities_mat(df, time_period_months):
    df['date'] = pd.to_datetime(df['date'])

    df[f'fs_m{time_period_months}'] = np.nan
    df[f'bs_m{time_period_months}'] = np.nan
    df[f'q_m{time_period_months}'] = np.nan


    embeddings_matrix = np.vstack(df['embedding'].values)  # (n_samples, m)
    norms = np.linalg.norm(embeddings_matrix, axis=1) 

    def calculate_for_row(i, df, time_period_months, embeddings_matrix, norms):
        curr_emb = embeddings_matrix[i]
        curr_norm = norms[i]
        curr_date = df.at[i, 'date']
        
        forward_time = curr_date + pd.DateOffset(months=time_period_months)
        backward_time = curr_date - pd.DateOffset(months=time_period_months)
        
        forward_idx = df[(df['date'] > curr_date) & (df['date'] <= forward_time)].index
        forward_embs = embeddings_matrix[forward_idx]  #(n_forward, embedding_dim)
        

        backward_idx = df[(df['date'] < curr_date) & (df['date'] >= backward_time)].index
        backward_embs = embeddings_matrix[backward_idx]  # (n_backward, embedding_dim)
        

        if len(forward_embs) > 0:
            forward_similarities = np.dot(forward_embs, curr_emb) / (norms[forward_idx] * curr_norm)
            forward_similarity = np.sum(forward_similarities)
        else:
            forward_similarity = 0

        if len(backward_embs) > 0:
            backward_similarities = np.dot(backward_embs, curr_emb) / (norms[backward_idx] * curr_norm)
            backward_similarity = np.sum(backward_similarities)
        else:
            backward_similarity = 0
        
        return forward_similarity, backward_similarity
    

    for i in range(len(df)):
        forward_sim, backward_sim = calculate_for_row(i, df, time_period_months, embeddings_matrix, norms)
        df.at[i, f'fs_m{time_period_months}'] = forward_sim
        df.at[i, f'bs_m{time_period_months}'] = backward_sim
        if backward_sim != 0:
            df.at[i, f'q_m{time_period_months}'] = forward_sim / backward_sim
        else:
            df.at[i, f'q_m{time_period_months}'] = np.nan
        
        print(forward_sim)
    
    return df

def main():
    df = pd.read_feather('../data/repo_embs.feather')
    df_12_sims = calculate_similarities_mat(df, 12)
    df_all_sims = calculate_similarities_mat(df_12_sims, 24)
    final_df = df_all_sims.drop(columns=['paper_url', 'date', 'topic', 'subfield', 'embedding'])
    final_df.to_feather('../data/fs_bs.feather')
    
    
if __name__=='__main__':
    main()