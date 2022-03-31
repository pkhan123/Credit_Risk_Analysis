# general custom functions
# import libraries
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier



# custom helper functions
def Custom_Helper_Module():
    print_statement = '''
    Available General Custom Functions: 

    Check_Missing_Values(input_dataset)
    Check_Feature_Details(input_dataset, input_feature)
    Create_Dummy_Variables(input_dataset, input_feature_list)
    Preliminary_Feature_Selection(input_X_train, input_y_train)
    Check_Correlation(input_X_train)
    Check_Multicollinearity(input_X_train, numerical_feature_list)
    Make_Feature_Selection(input_X_train, input_y_train, max_validation_round)
    Remove_Outlies(input_dataset, input_features)

    Convert_Datetime_To_Months(df, column)
    Convert_Loan_Tenure_To_Months(df, column)
    Convert_Employment_Length_To_Years(df, column)
    '''
    return print(print_statement)

# check missing values
def Check_Missing_Values(input_dataset):
    high_missing_value_columns = []
    max_missing_value_percentage = 80
    missing_data = input_dataset.isnull().sum()
    missing_data_percentage = (missing_data*100/len(input_dataset)).round(2) 
    data_Types = input_dataset.dtypes
    missing_data_df = pd.DataFrame({'Missing_Data': missing_data,
                                    'Missing_Data (%)': missing_data_percentage,
                                    'Data_Type': data_Types})
    for column in input_dataset:
        mdp = (input_dataset[column].isnull().sum()*100/len(input_dataset[column])).round(2)
        if mdp > max_missing_value_percentage: high_missing_value_columns.append(column)
    print('Following featues have more than 80% missing values: ', len(high_missing_value_columns))
    print(high_missing_value_columns)
    return missing_data_df


# replace missing categorical values
def Replace_Missing_Categorical_Values(input_dataset, input_feature_list):
    for input_feature in input_feature_list:
        fill_value = 'missing_value_'+input_feature
        input_dataset[input_feature] = input_dataset[input_feature].fillna(fill_value)



# replace missing numerical values
def Replace_Missing_Numerical_Values(input_dataset, input_feature_list):
    imputer_median = SimpleImputer(missing_values=np.nan, strategy='median')
    for input_feature in input_feature_list:
        imputer_median.fit(input_dataset[[input_feature]])
        input_dataset[[input_feature]] = imputer_median.transform(input_dataset[[input_feature]])

        

# check feature details
def Check_Feature_Details(input_dataset, input_feature):
    unique_features = input_dataset[input_feature].unique()
    print(unique_features)
    value_counts = input_dataset[input_feature].value_counts(ascending=False).head(10)
    value_counts_percentage = (value_counts*100/len(input_dataset)).round(2)
    feature_details_df = pd.DataFrame({'Value_Counts': value_counts,
                                    'Value_Counts (%)': value_counts_percentage})
    return feature_details_df


# reduce category
def Reduce_Category(input_dataset, input_feature_list):
    print('Reducing categories up to 10 (top) categories for each feature ...\n')
    for input_feature in input_feature_list:
        top_10_categories = [x for x in input_dataset[input_feature].value_counts().sort_values(ascending=False).head(9).index]
        input_dataset[input_feature] = np.where(input_dataset[input_feature].isin(top_10_categories), input_dataset[input_feature], 'Other')


# create dummy variables for all the categorical variables
def Create_Dummy_Variables(input_dataset, input_feature_list):
    print('Creating dummies for top 10 levels in each feature ...\n')
    # print('Creating dummies for top 10 levels in each feature having data more than 1% ...\n')
    # input_feature_list = input_dataset.columns.values.tolist()
    for input_feature in input_feature_list:
        # one_percent = len(input_dataset)/100
        top_10_values = [x for x in input_dataset[input_feature].value_counts().sort_values(ascending=False).head(10).index]
        # top_10_values_above_one_percent = [x for x in top_10_values if input_dataset[input_feature].value_counts()[x] > one_percent]
        print(top_10_values)
        # print(top_10_values_above_one_percent)
        for label in top_10_values:
            # if number of unique values is greater than 1 and less than 10
            # and level count greater than 1%
            if (1 < len(top_10_values) <= 10): 
                input_dataset[str(input_feature)+'_'+str(label)] = np.where(input_dataset[input_feature]==label, 1, 0)
            else:
                print('Dummy not created for:', input_feature)
    input_dataset = input_dataset.drop(input_feature_list, axis = 1, inplace=True)
    return input_dataset


# preliminary feature selection as per feature importance
def Preliminary_Feature_Selection(input_X_train, input_y_train):
    random_state = 42
    min_feature_importances = 0.001 # (default: 0.001)
    model = RandomForestClassifier(random_state = random_state)
    model.fit(input_X_train,input_y_train)
    feature_importances_pd = pd.Series(model.feature_importances_, index=input_X_train.columns)
    print(feature_importances_pd.nlargest(10))
    feature_dictionary = {} 
    for feature, importance in zip(input_X_train.columns, model.feature_importances_):
        feature_dictionary[feature] = importance 
    sorted_feature_dictionary = sorted(feature_dictionary.items(), key=lambda x:x[1], reverse = True)
    local_counter = 0
    local_sum = 0
    rearranged_feature_list = []
    for index, tuple in enumerate(sorted_feature_dictionary):
        important_feature = tuple[0]
        importances = tuple[1]
        # conditional loop to select number of features
        if importances > min_feature_importances: 
            local_counter = local_counter + 1
            local_sum = local_sum + importances
            rearranged_feature_list.append(important_feature)
        else:
            pass
    print('Random State: ', random_state, '\tSelected features: ', local_counter, '\tImportance: ', local_sum)
    with open('preliminary_feature_list.txt', 'w') as f:
        for item in rearranged_feature_list:
            f.write('%s\n' % item)


# check correlation
def Check_Correlation(input_X_train):
    max_correlation = 0.95
    X_train_feature_list = input_X_train.columns.values.tolist()
    correlation_matrix = input_X_train.corr().abs()
    upper_traingular_part = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool_))
    unstacked_upper_traingular_part = upper_traingular_part.unstack()
    sorted_unstacked_upper_traingular_part = unstacked_upper_traingular_part.sort_values(ascending=False).head(10)
    print('Top 10 correlated pairs ... \n')
    print(sorted_unstacked_upper_traingular_part) 
    highly_correlated_feature_list = [column for column in upper_traingular_part.columns if any(upper_traingular_part[column] > max_correlation)]
    correlation_qualified_feature_list = list((Counter(X_train_feature_list) - Counter(highly_correlated_feature_list)).elements())
    print('\nFeature dropped: ')
    print(highly_correlated_feature_list)
    with open('correlation_qualified_feature_list.txt', 'w') as f:
        for item in correlation_qualified_feature_list:
            f.write('%s\n' % item)
    print('Correlation qualified features: ', len(correlation_qualified_feature_list))


# check multicollinearity
def Check_Multicollinearity(input_X_train, numerical_feature_list):
    max_multicollinearity = 5 # 5 can be considered high
    check_value = 999
    feature_to_drop = []
    input_X_train_list = input_X_train.columns.values.tolist()
    local_X_train = input_X_train[numerical_feature_list].copy()
    local_X_train_list = local_X_train.columns.values.tolist()
    # scale the dataset
    standard_scaler = StandardScaler()
    local_X_train[local_X_train_list] = standard_scaler.fit_transform(local_X_train[local_X_train_list])
    while check_value > max_multicollinearity:
        vif_dataset = local_X_train[local_X_train_list].copy()
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(vif_dataset.values, i) for i in range(vif_dataset.shape[1])]
        vif['Features'] = vif_dataset.columns
        vif = vif.sort_values(by=['VIF'], ascending=False)
        first_row_VIF = vif['VIF'].iloc[0]
        first_row_feature = vif['Features'].iloc[0]
        check_value = first_row_VIF
        if first_row_VIF > max_multicollinearity: 
            print(first_row_VIF, first_row_feature)
            drop_list = [first_row_feature]
            feature_to_drop.append(first_row_feature)
            local_X_train_list = list((Counter(local_X_train_list) - Counter(drop_list)).elements())
        else:
            print('VIFs < 5 ... OK')
    print('\nFeature dropped: ')
    print(feature_to_drop)
    multicollinearity_qualified_feature_list = list((Counter(input_X_train_list) - Counter(feature_to_drop)).elements())
    with open('multicollinearity_qualified_feature_list.txt', 'w') as f:
        for item in multicollinearity_qualified_feature_list:
            f.write('%s\n' % item)
    print('Multicollinearity qualified features: ', len(multicollinearity_qualified_feature_list))
    return print(vif)


# make feature selection
def Make_Feature_Selection(input_X_train, input_y_train, max_validation_round):
    print('Freature selection using RandomForestClassifier ...')
    min_feature_importances = 0.01 # (default: 0.01)
    max_random_state = 100
    round_counter = 0
    execute_round = 1
    validation_round = 0
    list_of_important_feature_lists = []
    selected_feature_list = []

    for random_state in range(1, max_random_state+1, 1): # default: 10 random_state
        if (execute_round == 1 or validation_round < max_validation_round):
            round_counter = round_counter + 1
            print('\nRound Counter: ', round_counter)
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(input_X_train, input_y_train,
                                                                test_size = 0.2, random_state = random_state, stratify = input_y_train)
            X_train_feature_list = X_train.columns.values.tolist()

            model = RandomForestClassifier(random_state = random_state*random_state)
            model.fit(X_train,y_train)
            # print top 10 features
            feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
            if round_counter == 1: print('Top 10 features: ')
            if round_counter == 1: print(feature_importances.nlargest(10))
            feature_dictionary = {} # a dictionary to hold feature_name: feature_importance
            # lets make dictionary of features as per importance
            for feature, importance in zip(X_train.columns, model.feature_importances_):
                feature_dictionary[feature] = importance #add the name/value pair 
            sorted_feature_dictionary = sorted(feature_dictionary.items(), key=lambda x:x[1], reverse = True)
            # lets make a list of selected features
            local_counter = 0
            local_sum = 0
            important_feature_list = []
            for index, tuple in enumerate(sorted_feature_dictionary):
                important_feature = tuple[0]
                importances = tuple[1]
                # conditional loop to select number of features
                if importances > min_feature_importances: 
                    local_counter = local_counter + 1
                    local_sum = local_sum + importances
                    important_feature_list.append(important_feature)
                else:
                    pass
            
            list_of_important_feature_lists.append(important_feature_list)
            common_elements_in_all_lists = list(set.intersection(*map(set, list_of_important_feature_lists)))
            if set(selected_feature_list) != set(common_elements_in_all_lists):
                dropped_this_round = list((Counter(selected_feature_list) - Counter(common_elements_in_all_lists)).elements())
                selected_feature_list = common_elements_in_all_lists
                execute_round = 1
                validation_round = 0
            else:
                dropped_this_round = []
                execute_round = 0
                validation_round = validation_round + 1

            print('Random State: ', random_state, '\tSelected features: ', local_counter, '\tImportance: ', local_sum)
            if round_counter == 1: print(important_feature_list)
            print('Total common features up to this round:', len(selected_feature_list), '\tValidation Round:', validation_round)
            print('Dropped this round: ', dropped_this_round)
            if (execute_round == 1 or validation_round < max_validation_round):
                print('Execute Next Round: Yes')
            else:
                print('Execute Next Round: No')

        else:
            pass

    feature_to_remove = list((Counter(X_train_feature_list) - Counter(common_elements_in_all_lists)).elements())
    # input_X_train = input_X_train.drop(feature_to_remove, axis = 1, inplace=True)
    print('\nFeature dropped during final selection: ')
    list_of_all_important_features = list({x for l in list_of_important_feature_lists for x in l})
    important_feature_removed = list((Counter(list_of_all_important_features) - Counter(common_elements_in_all_lists)).elements())
    print(important_feature_removed)
    with open('selected_feature_list.txt', 'w') as f:
        for item in selected_feature_list:
            f.write('%s\n' % item)
    print('Final selected features:', len(selected_feature_list))
    print('Final list of selected features: ')
    print(selected_feature_list)


# remove outlies upto 0.27% (or fraction 0.0027)
def Remove_Outlies(input_dataset_X, input_dataset_y, input_features):
    outliers_removal_factor = 0.002700 # beyond 3 sigma for normal distribution
    total_data_removed = 0
    initial_data_length = len(input_dataset_X.index)
    numerical_feature_list = input_features
    all_removed_counter = 0
    all_removed_counter_percentage = 0
    all_feature_row_drop_list = []
    for numerical_feature in numerical_feature_list:
        data_removed = 0
        low_removed_counter = 0
        high_removed_counter = 0
        feature_row_drop_list = []
        initial_min_value = input_dataset_X[numerical_feature].min()
        initial_max_value = input_dataset_X[numerical_feature].max()
        input_dataset_range = (abs(initial_max_value - initial_min_value))
        initial_median_value = input_dataset_X[numerical_feature].median()
        if input_dataset_range != 0:
            min_cut = round(abs(initial_median_value - initial_min_value)*outliers_removal_factor/input_dataset_range, 6)
            max_cut = round(abs(initial_max_value - initial_median_value)*outliers_removal_factor/input_dataset_range, 6)
        else:
            min_cut = 0
            max_cut = 0
        # print('min_cut: ', min_cut, ' max_cut: ', max_cut)
        min_quantile = round(input_dataset_X[numerical_feature].quantile(min_cut), 6)
        max_quantile = round(input_dataset_X[numerical_feature].quantile(1-max_cut), 6)

        index_names_low = list(input_dataset_X[input_dataset_X[numerical_feature] < min_quantile ].index)
        low_removed_counter = len(index_names_low)
        
        index_names_high = list(input_dataset_X[input_dataset_X[numerical_feature] > max_quantile ].index)
        high_removed_counter = len(index_names_high)

        feature_row_drop_list = list(set(index_names_low + index_names_high))
        all_feature_row_drop_list = list(set(all_feature_row_drop_list + feature_row_drop_list))
        
        data_removed = len(feature_row_drop_list)
        total_data_removed = len(all_feature_row_drop_list)

        data_removed_percentage = round((data_removed*100/initial_data_length), 2)
        total_data_removed_percentage = round((total_data_removed*100/initial_data_length), 2)

        final_min_value = input_dataset_X[numerical_feature].min()
        final_max_value = input_dataset_X[numerical_feature].max()
        final_median_value = input_dataset_X[numerical_feature].median()

        print('Feature: ', numerical_feature)
        print('Initial Min: ', initial_min_value, ' Initial Median: ', initial_median_value, ' Initial Max: ', initial_max_value)
        print('Min Cut: ', min_quantile, ' Max Cut: ', max_quantile)
        print('Data Removed: ', data_removed, '(',data_removed_percentage,'%)'
            ' Total Data Removed: ', total_data_removed, '(',total_data_removed_percentage,'%)')
        print('Low Value Removed: ', low_removed_counter, ' High Value Removed: ', high_removed_counter)
        print('Final Min: ', final_min_value, ' Final Median: ', final_median_value, ' Final Max: ', final_max_value)
        print('\n')
    input_dataset_X.drop(all_feature_row_drop_list, inplace = True)
    input_dataset_y.drop(all_feature_row_drop_list, inplace = True)
    all_removed_counter = len(all_feature_row_drop_list)
    all_removed_counter_percentage = round((all_removed_counter*100/initial_data_length), 2)
    print('Total outlier removed: ', all_removed_counter, '(',all_removed_counter_percentage,'%)')
    print('\n')


# convert datetime columns to month
def Convert_Datetime_To_Months(df, column):
    today_date = pd.to_datetime('2020-12-31') # we have data upto December 2014
    df[column] = pd.to_datetime(df[column], format = '%b-%y')
    df['months_since_' + column] = round(pd.to_numeric((today_date - df[column]) / np.timedelta64(1, 'M')))
    df['months_since_' + column] = df['months_since_' + column].apply(lambda x: df['months_since_' + column].max() if x < 0 else x)
    df.drop(columns = [column], inplace = True)


# convert loan tenure to months
def Convert_Loan_Tenure_To_Months(df, column):
    df[column] = pd.to_numeric(df[column].str.replace(' months', ''))


# convert emploment length to year
def Convert_Employment_Length_To_Years(df, column):
    df[column] = df[column].str.replace(r'\+ years', '', regex=True)
    df[column] = df[column].str.replace('< 1 year', str(0))
    df[column] = df[column].str.replace(' years', '')
    df[column] = df[column].str.replace(' year', '')
    df[column] = pd.to_numeric(df[column])


