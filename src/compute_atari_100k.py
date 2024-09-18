import os
import os.path as osp
import json
import numpy as np
import pandas as pd
from rliable import metrics

ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]

RANDOM_SCORES = {
 'Alien': 227.8,
 'Amidar': 5.8,
 'Assault': 222.4,
 'Asterix': 210.0,
 'BankHeist': 14.2,
 'BattleZone': 2360.0,
 'Boxing': 0.1,
 'Breakout': 1.7,
 'ChopperCommand': 811.0,
 'CrazyClimber': 10780.5,
 'DemonAttack': 152.1,
 'Freeway': 0.0,
 'Frostbite': 65.2,
 'Gopher': 257.6,
 'Hero': 1027.0,
 'Jamesbond': 29.0,
 'Kangaroo': 52.0,
 'Krull': 1598.0,
 'KungFuMaster': 258.5,
 'MsPacman': 307.3,
 'Pong': -20.7,
 'PrivateEye': 24.9,
 'Qbert': 163.9,
 'RoadRunner': 11.5,
 'Seaquest': 68.4,
 'UpNDown': 533.4
}

HUMAN_SCORES = {
 'Alien': 7127.7,
 'Amidar': 1719.5,
 'Assault': 742.0,
 'Asterix': 8503.3,
 'BankHeist': 753.1,
 'BattleZone': 37187.5,
 'Boxing': 12.1,
 'Breakout': 30.5,
 'ChopperCommand': 7387.8,
 'CrazyClimber': 35829.4,
 'DemonAttack': 1971.0,
 'Freeway': 29.6,
 'Frostbite': 4334.7,
 'Gopher': 2412.5,
 'Hero': 30826.4,
 'Jamesbond': 302.8,
 'Kangaroo': 3035.0,
 'Krull': 2665.5,
 'KungFuMaster': 22736.3,
 'MsPacman': 6951.6,
 'Pong': 14.6,
 'PrivateEye': 69571.3,
 'Qbert': 13455.0,
 'RoadRunner': 7845.0,
 'Seaquest': 42054.7,
 'UpNDown': 11693.2
}

MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)

def score_normalization(res_dict, min_scores, max_scores):
  games = res_dict.keys()
  norm_scores = {}
  for game, scores in res_dict.items():
    norm_scores[game] = (scores - min_scores[game])/(max_scores[game] - min_scores[game])
  return norm_scores

def convert_to_matrix(score_dict):
   keys = sorted(list(score_dict.keys()))
   return np.stack([score_dict[k] for k in keys], axis=1)

def create_score_dict_atari_100k(main_df, normalization=True,
                                 evaluation_key = 'eval_average_return'):
  """Creates a dictionary of scores."""
  score_dict = {}
  for key, df in main_df.items():
    score_dict[key] = df[evaluation_key].values
  if normalization:
    score_dict = score_normalization(score_dict, RANDOM_SCORES, HUMAN_SCORES)
  return score_dict

def get_scores(df, normalization=True, eval='Final'):
  score_dict_df = create_score_dict_atari_100k(df, normalization=normalization)
  score_matrix = convert_to_matrix(score_dict_df)
  median, mean = MEDIAN(score_matrix), MEAN(score_matrix)
  print('{}: Median: {}, Mean: {}'.format(eval, median, mean))
  return score_dict_df, score_matrix

def save_df(df_to_save, name, base_df_path='atari_100k'):
  base_dir = osp.join(base_df_path, name)
  os.makedirs(base_dir)
  for game in df_to_save.keys():
    file_name = osp.join(base_dir, f'{game}.json')
    with open(file_name, 'w') as f:
      df_to_save[game].to_json(f, orient='records')
    print(f'Saved {file_name}')

def read_df(name, base_df_path='atari_100k'):
  base_dir = osp.join(base_df_path, name)
  df_to_read = {}
  for game in ATARI_100K_GAMES:
    file_name = osp.join(base_dir, f'{game}.json')
    with open(file_name, 'r') as f:
      df_to_read[game] = pd.read_json(f, orient='records')
  return df_to_read

def remove_additional_evaluations(df_to_filter):
  new_df = {}
  for key in df_to_filter.keys():
    df_a = df_to_filter[key]
    df_a = df_a[df_a.index == '0'].drop(['iteration'], axis=1)
    new_df[key] = df_a
  return new_df

def load_and_read_scores(algorithm_name, num_evals=None):
  print(f'Loading scores for {algorithm_name}:')
  df = read_df(algorithm_name)
  if num_evals is None:
    return get_scores(df)
  # Read multiple evals.
  final_scores_df, max_scores_df = {}, {}
  for game, game_df in df.items():
    final_scores_df[game] = game_df[game_df['iteration'] == num_evals-1]
    max_scores_df[game] = game_df.groupby('run_number').max()
  return get_scores(final_scores_df), get_scores(max_scores_df, eval='Max')

def read_curl_scores():
  print(f'Loading scores for CURL:')
  df = pd.read_json('atari_100k/CURL_10_evals.json', orient='records')
  score_dict = {'Max': {}, 'Final': {}}
  for game in ATARI_100K_GAMES:
    game_df = df[df['game'] == game]
    score_dict['Final'][game] = game_df['HNS'].values
    score_dict['Max'][game] = game_df['Max HNS'].values
  score_matrices = {}
  for key, val in score_dict.items():
    score_matrices[key] = convert_to_matrix(val)
    median, mean = MEDIAN(score_matrices[key]), MEAN(score_matrices[key])
    print('{}: Median: {}, Mean: {}'.format(key, median, mean))
  return (score_dict['Final'], score_matrices['Final']), (
      score_dict['Max'], score_matrices['Max'])

def load_json_scores(algorithm_name, base_path='atari_100k'):
  print(f'Loading scores for {algorithm_name}:')
  path = osp.join(base_path, f'{algorithm_name}.json')
  with open(path, 'r') as f:
    scores = json.load(f)
  scores = {game: np.array(val) for game, val in scores.items()}
  scores = score_normalization(scores, RANDOM_SCORES, HUMAN_SCORES)
  score_matrix = convert_to_matrix(scores)
  median, mean = MEDIAN(score_matrix), MEAN(score_matrix)
  print('{}: Median: {}, Mean: {}'.format(eval, median, mean))
  return scores, score_matrix

def compute_atari_100k():
  (score_dict_der, score_der), (_, score_der_max) = load_and_read_scores(
      'DER', num_evals=10)
  (score_dict_curl, score_curl), (_, score_curl_max) = read_curl_scores()
  
  score_dict_otr, score_otr = load_json_scores('OTRainbow')
  score_dict_drq, score_drq = load_json_scores('DrQ')
  score_dict_spr, score_spr = load_json_scores('SPR')
  score_dict_simple, score_simple = load_json_scores('SimPLe')
  # DrQ agent but with standard epsilon values of 0.01/0.001 for training
  # and evaluation eps-greedy parameters
  score_dict_drq_eps, score_drq_eps = load_json_scores('DrQ(eps)')
  
  score_data_dict = {'CURL': score_curl,
                      'DrQ': score_drq,
                      'DrQ(ε)': score_drq_eps,
                      'DER': score_der,
                      'SimPLe': score_simple,
                      'OTR': score_otr,
                      'SPR': score_spr}

  scores = {'CURL': score_dict_curl,
             'DrQ': score_dict_drq,
             'DrQ(ε)': score_dict_drq_eps,
             'DER': score_dict_der,
             'SimPLe': score_dict_simple,
             'OTR': score_dict_otr,
             'SPR': score_dict_spr}
  
  for method in scores.keys():
      for game in scores[method].keys():
          mean_score = np.mean(scores[method][game])
          scores[method][game] = mean_score
  
  scores = pd.DataFrame(scores)

  return scores

if __name__ == '__main__':
  scores = compute_atari_100k().round(2)
  print(scores)