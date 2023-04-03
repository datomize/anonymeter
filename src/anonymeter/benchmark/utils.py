import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

from anonymeter.evaluators import SinglingOutEvaluator
from anonymeter.evaluators import LinkabilityEvaluator
from anonymeter.evaluators import InferenceEvaluator


RANDOM_SEED = 42
PRIVACY = 'privacy_lv'
MEAN = 'mean'
ROUND_N = 5
COLUMN_LIST = ['privacy_lv', 'risk', 'risk_ci', 'main_attack', 'main_attack_er', 'base_attack', 'base_attack_er',
               'control_attack', 'control_attack_er']


def run_singling_out(synthetic_df, original_df, control_df, n_attacks, mode='univariate'):
    evaluator = SinglingOutEvaluator(ori=original_df,
                                     syn=synthetic_df,
                                     control=control_df,
                                     n_attacks=n_attacks)

    try:
        evaluator.evaluate(mode=mode)
        return evaluator

    except RuntimeError as ex:
        print(f"Singling out evluation failed with {ex}. Please re-run this cell."
              "For more stable results increase `n_attacks`. Note that this will "
              "make the evaluation slower.")


def run_linkability(synthetic_df, original_df, control_df, n_attacks, aux_cols, n_neighbors):

    evaluator = LinkabilityEvaluator(ori=original_df,
                                     syn=synthetic_df.astype(original_df.dtypes),
                                     control=control_df,
                                     n_attacks=n_attacks,
                                     aux_cols=aux_cols,
                                     n_neighbors=n_neighbors)

    evaluator.evaluate(n_jobs=-2)  # n_jobs follow joblib convention. -1 = all cores, -2 = all execept one
    return evaluator


def run_inference(item, original_df, control_df, n_attacks, save_path, plot=False):
    synthetic_df, name = item[1], item[0]
    columns = original_df.columns
    results = []

    worse_val = ('None', 0.0)
    worst_eval = None

    for secret in columns:

        aux_cols = [col for col in columns if col != secret]

        evaluator = InferenceEvaluator(ori=original_df,
                                       syn=synthetic_df,
                                       control=control_df,
                                       aux_cols=aux_cols,
                                       secret=secret,
                                       n_attacks=n_attacks)
        evaluator.evaluate(n_jobs=-2)
        results.append((secret, evaluator.results()))
        if evaluator.results().attack_rate[0] > worse_val[1]:
            worse_val = (secret, evaluator.results().attack_rate[0])
            worst_eval = evaluator

    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    risks = [res[1].risk().value for res in results]
    columns = [res[0] for res in results]

    plt.bar(x=columns, height=risks, alpha=0.5, ecolor='black', capsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Measured inference risk")
    plt.xlabel("Secret column")
    plt.savefig(save_path + f'{name}.png')
    if plot:
        plt.show()
    plt.close()

    return worst_eval, worse_val[0]


def round_n(num, n):
    if num != 0.0:
        return num.round(n)
    else:
        return num


def most_freq(ls):
    return max(set(ls), key=ls.count)


def split_data(df, n_chunks, shuffled=False):
    batch_size = int(df.shape[0]/n_chunks)
    df_list = []
    if shuffled:
        lst = np.arange(df.shape[0])
        random.Random(RANDOM_SEED).shuffle(lst)
        for i in range(n_chunks):
            df_list.append(df.loc[lst[i * batch_size: (i + 1) * batch_size], :].reset_index(drop=True))
    else:
        for i in range(n_chunks):
            df_list.append(df.loc[i * batch_size: (i + 1) * batch_size, :].reset_index(drop=True))
    return df_list



def evaluate_evaluators(indices, evaluators, type_eval):
    updated_columns = COLUMN_LIST.copy()
    updated_columns = [f"{type_eval}_" + val if val != PRIVACY else val for val in updated_columns]
    risks = []
    risks_ci = []
    main_attack = []
    main_attack_er = []
    base_attack = []
    base_attack_er = []
    control_attack = []
    control_attack_er = []
    for evaluator in evaluators:
        if evaluator is not None:
            risk = evaluator.risk()
            result = evaluator.results()
            risks.append(risk.value)
            risks_ci.append(risk.ci)
            main_attack.append(result.attack_rate.value)
            main_attack_er.append(result.attack_rate.error)
            base_attack.append(result.baseline_rate.value)
            base_attack_er.append(result.baseline_rate.error)
            control_attack.append(result.control_rate.value)
            control_attack_er.append(result.control_rate.error)
        else:
            risks.append(np.NAN)
            risks_ci.append((np.NAN, np.NAN))
            main_attack.append(np.NAN)
            main_attack_er.append(np.NAN)
            base_attack.append(np.NAN)
            base_attack_er.append(np.NAN)
            control_attack.append(np.NAN)
            control_attack_er.append(np.NAN)

    risks.append(round_n(np.nanmean(risks), ROUND_N))
    ci_means = np.nanmean(risks_ci, axis=0)
    risks_ci.append((round_n(ci_means[0], ROUND_N), round_n(ci_means[1], ROUND_N)))
    main_attack.append(round_n(np.nanmean(main_attack), ROUND_N))
    main_attack_er.append(round_n(np.nanmean(main_attack_er), ROUND_N))
    base_attack.append(round_n(np.nanmean(base_attack), ROUND_N))
    base_attack_er.append(round_n(np.nanmean(base_attack_er), ROUND_N))
    control_attack.append(round_n(np.nanmean(control_attack), ROUND_N))
    control_attack_er.append(round_n(np.nanmean(control_attack_er), ROUND_N))

    df = pd.DataFrame({updated_columns[i]: v for i, v in enumerate(
        [indices, risks, risks_ci, main_attack, main_attack_er, base_attack, base_attack_er, control_attack,
         control_attack_er])}
                      )
    return df

