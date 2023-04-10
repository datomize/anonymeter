import os
import pandas as pd
from pathos.pools import ProcessPool
from functools import partial
from utils import run_singling_out, run_linkability, run_inference, evaluate_evaluators, split_data, \
    most_freq, MEAN, PRIVACY


def run_singling_out_uni_eval(data_path, res_path, ori_path, control_path, name, n_attacks, n_chunks):
    data_map = {f'dat_{name}_{i}': v for i, v in enumerate(split_data(pd.read_csv(data_path), n_chunks, shuffled=True))}

    # Run SinglingOut univariate
    indices = list(data_map.keys()) + [MEAN]
    f_name = res_path + f"singlingout_uni_res_{name}.csv"

    if os.path.exists(f_name):
        df_results = pd.read_csv(f_name)
    else:
        df_results = pd.DataFrame(indices, columns=[PRIVACY])

    ori_df = pd.read_csv(ori_path)
    control_df = pd.read_csv(control_path)

    for n_attack in n_attacks:
        with ProcessPool() as pool:
            evaluators = pool.map(
                partial(run_singling_out, original_df=ori_df, control_df=control_df, n_attacks=n_attack,
                        mode='univariate'),
                list(data_map.values()))
        df_results = df_results.merge(evaluate_evaluators(indices=list(data_map.keys()) + [MEAN], evaluators=evaluators,
                                                          type_eval=f'singlingout_uni_{n_attack}'), how='left',
                                      on=PRIVACY)

    df_results.to_csv(f_name, index=False)
    mean = df_results[df_results[PRIVACY] == MEAN]
    mean[PRIVACY] = name
    return mean


def run_singling_out_multi_eval(data_path, res_path, ori_path, control_path, name, n_attacks, n_cols, n_chunks):
    data_map = {f'dat_{name}_{i}': v for i, v in enumerate(split_data(pd.read_csv(data_path), n_chunks, shuffled=True))}
    # Run SinglingOut multivariate
    indices = list(data_map.keys()) + [MEAN]
    f_name = res_path + f"singlingout_multi_res_{name}.csv"

    if os.path.exists(f_name):
        df_results = pd.read_csv(f_name)
    else:
        df_results = pd.DataFrame(indices, columns=[PRIVACY])

    ori_df = pd.read_csv(ori_path)
    control_df = pd.read_csv(control_path)
    for n_attack in n_attacks:
        for n_col in n_cols:
            with ProcessPool() as pool:
                evaluators = pool.map(
                    partial(run_singling_out, original_df=ori_df, control_df=control_df, n_attacks=n_attack,
                            n_cols=n_col, mode='univariate'),
                    list(data_map.values()))
            df_results = df_results.merge(
                evaluate_evaluators(indices=list(data_map.keys()) + [MEAN], evaluators=evaluators,
                                    type_eval=f'singlingout_multi_{n_col}_{n_attack}'), how='left', on=PRIVACY)

    df_results.to_csv(f_name, index=False)
    mean = df_results[df_results[PRIVACY] == MEAN]
    mean[PRIVACY] = name
    return mean


def run_linkability_eval(data_path, res_path, ori_path, control_path, name, n_attacks, aux_cols, n_neighbors, n_chunks):
    data_map = {f'dat_{name}_{i}': v for i, v in enumerate(split_data(pd.read_csv(data_path), n_chunks, shuffled=True))}

    # Run Linkability
    indices = list(data_map.keys()) + [MEAN]
    f_name = res_path + f"linkability_res_{name}.csv"

    if os.path.exists(f_name):
        df_results = pd.read_csv(f_name)
    else:
        df_results = pd.DataFrame(indices, columns=[PRIVACY])

    ori_df = pd.read_csv(ori_path)
    control_df = pd.read_csv(control_path)

    for n_attack in n_attacks:
        n_attack = control_df.shape[0] if n_attack > control_df.shape[0] else n_attack
        for n_neighbor in n_neighbors:
            a = run_linkability(list(data_map.values())[0], original_df=ori_df, control_df=control_df, n_attacks=n_attack,
                            aux_cols=aux_cols, n_neighbors=n_neighbor)
            with ProcessPool() as pool:
                evaluators = pool.map(
                    partial(run_linkability, original_df=ori_df, control_df=control_df, n_attacks=n_attack,
                            aux_cols=aux_cols, n_neighbors=n_neighbor),
                    list(data_map.values()))
            df_results = df_results.merge(
                evaluate_evaluators(indices=list(data_map.keys()) + [MEAN], evaluators=evaluators,
                                    type_eval=f'linkability_{n_neighbor}_{n_attack}'), how='left', on=PRIVACY)

    df_results.to_csv(f_name, index=False)
    mean = df_results[df_results[PRIVACY] == MEAN]
    mean[PRIVACY] = name
    return mean


def run_inference_eval(data_path, res_path, ori_path, control_path, name, n_attacks, n_chunks):
    data_map = {f'dat_{name}_{i}': v for i, v in enumerate(split_data(pd.read_csv(data_path), n_chunks, shuffled=True))}

    # Run Inference
    indices = list(data_map.keys()) + [MEAN]
    f_name = res_path + f"inference_res_{name}.csv"
    save_fig_path = res_path + f'inference_plots/dat_{name}/'
    os.makedirs(save_fig_path, exist_ok=True)
    if os.path.exists(f_name):
        df_results = pd.read_csv(f_name)
    else:
        df_results = pd.DataFrame(indices, columns=[PRIVACY])

    ori_df = pd.read_csv(ori_path)
    control_df = pd.read_csv(control_path)
    for n_attack in n_attacks:
        n_attack = control_df.shape[0] if n_attack > control_df.shape[0] else n_attack
        with ProcessPool() as pool:
            evaluators, worst = zip(*pool.map(
                partial(run_inference, original_df=ori_df, control_df=control_df, n_attacks=n_attack,
                        save_path=save_fig_path, plot=False), list(data_map.items())))
        df_results = df_results.merge(
            evaluate_evaluators(indices=list(data_map.keys()) + [MEAN], evaluators=evaluators,
                                type_eval=f'inference_{n_attack}'), how='left', on=PRIVACY)
        df_results[f'inference_{n_attack}_worst_col'] = list(worst) + [most_freq(list(worst))]

    df_results.to_csv(f_name, index=False)
    mean = df_results[df_results[PRIVACY] == MEAN]
    mean[PRIVACY] = name
    return mean


def get_aux_columns(table_name):
    aux_cols = None
    if table_name == 'adults':
        aux_cols = [['type_employer', 'education', 'hr_per_week', 'capital_loss', 'capital_gain'],
                    ['race', 'sex', 'fnlwgt', 'age', 'country']]
    elif table_name == 'cardio':
        aux_cols = [['HEIGHT', 'WEIGHT', 'AP_HIGH', 'AP_LOW'],
                    ['AGE', 'GENDER', 'CHOLESTEROL', 'GLUCOSE', 'SMOKE', 'ALCOHOL', 'PHYSICAL_ACTIVITY',
                     'CARDIO_DISEASE']]
    elif table_name == 'house_sale':
        aux_cols = [['id', 'date', 'price', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'],
                    ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
                     'grade', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]
    elif table_name == 'german_credit':
        aux_cols = [['Age', 'Sex', 'Job'],
                    ['Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Risk']]
    elif table_name == 'stroke':
        aux_cols = [['gender', 'age', 'stroke', 'smoking_status'],
                    ['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                     'avg_glucose_level', 'bmi']]
    elif table_name == 'startup':
        aux_cols = [['state_code', 'latitude', 'longitude', 'city', 'founded_at', 'first_funding_at',
                     'category_code'],
                    ['closed_at', 'last_funding_at', 'age_first_funding_year', 'age_last_funding_year',
                     'age_first_milestone_year', 'age_last_milestone_year', 'relationships', 'funding_rounds',
                     'funding_total_usd', 'milestones', 'state_code', 'is_CA', 'is_NY', 'is_MA', 'is_TX',
                     'is_otherstate', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate', 'category_code',
                     'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo',
                     'is_ecommerce', 'is_biotech', 'is_othercategory', 'has_VC', 'has_angel', 'has_roundA',
                     'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants', 'is_top500', 'status']]
    elif table_name == 'alphalytica':
        aux_cols = [['identifier', 'identifier_type', 'time_zone_name', 'classification', 'duration', 'source_id'],
                    ['centroid_lat', 'centroid_lon', 'country_code', 'province', 'ip_address', 'bump_count',
                     'home_country_code']]

    else:
        # assert, no
        print(f"No aux columns for table {table_name}")

    return aux_cols


def run_files(original_df_path, control_df_path, synt_path, res_path, **kwargs):
    if os.path.isdir(synt_path):
        files_dict = {f"dat_{f_name[:-4]}": synt_path + f"/{f_name}" for f_name in os.listdir(synt_path + '/') if
                      ".csv" in f_name}
    else:
        files_dict = {f"dat_{os.path.basename(synt_path)[:-4]}": synt_path}

    total_res_file = res_path + f'results.csv'
    if os.path.exists(total_res_file):
        results = pd.read_csv(total_res_file)
    else:
        results = pd.DataFrame(list(files_dict.keys()), columns=[PRIVACY])

    uni_attacks = kwargs.get("uni_attacks", [5000])
    n_chunks = kwargs.get("n_chunks", 1)
    results = results.combine_first(pd.concat([run_singling_out_uni_eval(data_path=p, res_path=res_path, name=n,
                                                                         n_attacks=uni_attacks,
                                                                         ori_path=original_df_path,
                                                                         control_path=control_df_path,
                                                                         n_chunks=n_chunks) for n, p in
                                               files_dict.items()]).reset_index(drop=True))
    results.to_csv(total_res_file, index=False)

    multi_attacks = kwargs.get("multi_attacks", [100])
    multi_cols = kwargs.get("multi_cols", [2, 3, 4, 5])
    results = results.combine_first(pd.concat(
        [run_singling_out_multi_eval(data_path=p, res_path=res_path, name=n, n_attacks=multi_attacks, n_cols=multi_cols,
                                     ori_path=original_df_path, control_path=control_df_path, n_chunks=n_chunks)
         for n, p in files_dict.items()]).reset_index(drop=True))
    results.to_csv(total_res_file, index=False)

    linkability_aux_cols = kwargs.get("aux_cols", None)
    linkability_attacks = kwargs.get("linkability_attacks", [5000])
    linkability_neighbors = kwargs.get("linkability_neighbors", [10])
    if linkability_aux_cols is not None:
        results = results.combine_first(pd.concat([run_linkability_eval(data_path=p, res_path=res_path, name=n,
                                                                        n_attacks=linkability_attacks,
                                                                        aux_cols=linkability_aux_cols,
                                                                        n_neighbors=linkability_neighbors,
                                                                        ori_path=original_df_path,
                                                                        control_path=control_df_path,
                                                                        n_chunks=n_chunks) for n, p in
                                                   files_dict.items()]).reset_index(drop=True))
        results.to_csv(total_res_file, index=False)

    inference_attacks = kwargs.get("inference_attacks", [5000])
    results = results.combine_first(pd.concat(
        [run_inference_eval(data_path=p, res_path=res_path, name=n, n_attacks=inference_attacks,
                            ori_path=original_df_path, control_path=control_df_path, n_chunks=n_chunks) for n, p in
         files_dict.items()]).reset_index(drop=True))
    results.to_csv(total_res_file, index=False)


def main():
    table_name = 'cardio'
    original_path = f"/home/lovakap/new_space/dataflow/benchmark_datasets/privacy_eval/{table_name}/train/{table_name}_train.csv"
    control_path = f"/home/lovakap/new_space/dataflow/benchmark_datasets/privacy_eval/{table_name}/control/{table_name}_control.csv"

    # Synth Path can either be single csv file or can be dir, in case of dir the attacks will be executed on all csvs
    synth_path = f"/home/lovakap/new_space/dataflow/benchmark_datasets/privacy_eval/{table_name}/synth"
    result_path = f"/home/lovakap/privacy_benchmark/{table_name}2/"

    os.makedirs(result_path, exist_ok=True)

    running_args = {
        'table_name': table_name,
        'n_chunks': 10,
        'shuffle': False,  # Shuffle data if n_chunks > 1
        'uni_attacks': [5000],
        'multi_attacks': [100],
        'multi_cols': [2, 3, 4, 5],
        'linkability_attacks': [5000],
        'aux_cols': get_aux_columns(table_name),  # for new datasets you have to add it manually
        'inference_attacks': [5000]
    }

    run_files(original_df_path=original_path, control_df_path=control_path, synt_path=synth_path, res_path=result_path,
              **running_args)


if __name__ == "__main__":
    main()
