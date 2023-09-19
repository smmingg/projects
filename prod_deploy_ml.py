import pandas as pd
import numpy as np
import featuretools as ft
import pickle
from flask import Flask, jsonify , request
import datetime

app = Flask(__name__)

# Load the pre-trained model from the pickle file
with open('xgb_selected.pkl', 'rb') as model_file:
    load_model = pickle.load(model_file)
    
# # Load the trained model
# load_model = pickle.load(open("xgb_selected.pkl", 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    df = request.get_json()  # If sending data via POST request
    df = pd.DataFrame(df)

  
    new_columns = ['member_id','day 1','day 2','day 3','day 4','day 5','day 6','day 7','day 8','day 9','day 10',
                       'day 11','day 12','day 13']

    results = []

    current_date = datetime.datetime.now().date()

    end_date = current_date - datetime.timedelta(days=2)
    start_date = end_date - datetime.timedelta(days=12)
    date_ranges = [(start_date , end_date)]

    def count_pos_neg(x):
        num_positive = len(x[x > 0])
        num_negative = len(x[x < 0])
        num_draw = len(x[x == 0])
        return pd.Series({'num_positive': num_positive, 'num_negative': num_negative, 'num_draw': num_draw})        

    def calculate_streaks(row):
        win_streak = 0
        longest_win_streak = 0
        lose_streak = 0
        longest_lose_streak = 0
        draw_streak = 0
        longest_draw_streak = 0

        for result in row[1:]:
            if result > 0:  # Check if value is positive
                win_streak += 1
                lose_streak = 0
                draw_streak = 0
                longest_win_streak = max(longest_win_streak, win_streak)
            elif result < 0:  # Check if value is negative
                lose_streak += 1
                win_streak = 0
                draw_streak = 0
                longest_lose_streak = max(longest_lose_streak, lose_streak)
            elif result == 0:  # Check if value is zero
                draw_streak += 1
                win_streak = 0
                lose_streak = 0
                longest_draw_streak = max(longest_draw_streak, draw_streak)

        return pd.Series([longest_win_streak, longest_lose_streak, longest_draw_streak])

    for start_date, end_date in date_ranges:
        df['working_date'] = pd.to_datetime(df['working_date'])
        new_df = df[(df['working_date'].dt.date >= start_date) & (df['working_date'].dt.date <= end_date)].copy()

        bet = new_df.groupby(['member_id']).agg({'bet_base': 'sum', 'bet_id': 'count', 'winlose': 'sum'}).reset_index()
        bet.rename(columns={'bet_base': 'bet_amount', 'bet_id': 'bet_count'}, inplace=True)

        result = new_df.groupby(['member_id', 'working_date'])['winlose'].sum().unstack(fill_value=0)
        result.columns = [str(col) for col in result.columns]  
        result = result.reset_index() 
        for col_name in result.columns:
            if col_name != 'member_id':
                parts = col_name.split()
                date_part = parts[0]
                result.rename(columns={col_name: date_part}, inplace=True)
        # Generate the list of date strings
        date_strings = [(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(13)]  
        # Create a list of column names, including 'member_id'
        column_names = ['member_id'] + date_strings
        missing_columns = [col for col in column_names if col not in result.columns]
        for col in missing_columns:
            result[col] = 0
        # Reorder the columns based on the specified order
        result = result[column_names]    
        result.columns = new_columns    

        # Incorporate the count_pos_neg function here
        pos_neg_counts = new_df.groupby('member_id')['winlose'].apply(count_pos_neg).unstack(fill_value=0).reset_index()
        pos_neg_counts.rename(columns={'num_positive': 'win_count', 'num_negative': 'lose_count', 'num_draw': 'draw_count'}, inplace=True)

        merged_result = pd.merge(result, bet, on='member_id', how='left')
        merged_result = pd.merge(merged_result, pos_neg_counts, on='member_id', how='left')


        results.append(merged_result)

    # Concatenate all DataFrames in the 'results' list to create the final DataFrame
    merged_df = pd.concat(results, ignore_index=True)
    merged_df['winning_margin'] = (merged_df['winlose'] / merged_df['bet_amount']) * 100
    merged_df['winrate'] = merged_df['win_count'] / (merged_df['win_count'] + merged_df['lose_count'] + merged_df['draw_count'])
    merged_df['loserate'] = merged_df['lose_count'] / (merged_df['win_count'] + merged_df['lose_count'] + merged_df['draw_count'])
    merged_df['drawrate'] = merged_df['draw_count'] / (merged_df['win_count'] + merged_df['lose_count'] + merged_df['draw_count'])
    streaks_df = merged_df.apply(calculate_streaks, axis=1)
    merged_df = merged_df.join(streaks_df)
    merged_df.rename(columns={0: 'win_streak', 1: 'lose_streak', 2: 'draw_streak'}, inplace=True)


    merged_df = merged_df.drop(['member_id'], axis=1)

    # Create an entity set
    es = ft.EntitySet(id='merged_df')
    # Add the dataframe to the entity set
    es.add_dataframe(dataframe_name='merged_df', dataframe= merged_df, index='index')

    # Run deep feature synthesis with transformation primitives
    feature_matrix, feature_defs = ft.dfs(entityset = es, target_dataframe_name = 'merged_df', trans_primitives = ['add_numeric', 'multiply_numeric'])
    # Select the feature columns from the dataframe
    X = feature_matrix

    selected_columns = [ "day 3", "day 6", "day 8", "bet_amount", "draw_count", "winrate", "loserate", "win_streak", "lose_streak", "draw_streak", "bet_amount + day 1", "bet_count + day 10", "bet_count + day 12", "bet_count + day 5", "bet_count + draw_streak", "bet_count + loserate", "bet_count + winning_margin", "day 1 + day 10", "day 1 + day 11", "day 1 + day 2", "day 1 + day 3", "day 1 + day 4", "day 1 + day 5", "day 1 + day 6", "day 1 + day 7", "day 1 + day 9", "day 1 + draw_count", "day 1 + draw_streak", "day 1 + lose_streak", "day 1 + win_streak", "day 10 + day 11", "day 10 + day 13", "day 10 + day 2", "day 10 + day 3", "day 10 + day 5", "day 10 + day 7", "day 10 + day 8", "day 10 + lose_count", "day 10 + loserate", "day 11 + day 12", "day 11 + day 13", "day 11 + day 2", "day 11 + day 3", "day 11 + day 5", "day 11 + day 6", "day 11 + day 7", "day 11 + day 8", "day 11 + day 9", "day 11 + draw_count", "day 11 + draw_streak", "day 11 + lose_count", "day 12 + day 13", "day 12 + day 2", "day 12 + day 4", "day 12 + day 6", "day 12 + day 7", "day 12 + day 8", "day 12 + day 9", "day 12 + draw_streak", "day 12 + lose_streak", "day 12 + loserate", "day 12 + winrate", "day 13 + day 3", "day 13 + day 4", "day 13 + day 6", "day 13 + day 7", "day 13 + day 8", "day 13 + day 9", "day 13 + draw_count", "day 13 + draw_streak", "day 13 + lose_count", "day 13 + lose_streak", "day 13 + win_count", "day 13 + win_streak", "day 13 + winlose", "day 13 + winning_margin", "day 2 + day 3", "day 2 + day 4", "day 2 + day 5", "day 2 + day 6", "day 2 + day 7", "day 2 + day 8", "day 2 + day 9", "day 2 + draw_count", "day 2 + draw_streak", "day 2 + lose_count", "day 2 + win_streak", "day 2 + winning_margin", "day 3 + day 4", "day 3 + day 6", "day 3 + day 7", "day 3 + day 8", "day 3 + day 9", "day 3 + draw_count", "day 3 + loserate", "day 3 + win_streak", "day 4 + day 6", "day 4 + day 8", "day 4 + day 9", "day 4 + lose_count", "day 4 + winlose", "day 4 + winning_margin", "day 5 + day 6", "day 5 + day 7", "day 5 + day 8", "day 5 + day 9", "day 5 + winning_margin", "day 6 + day 7", "day 6 + day 8", "day 6 + day 9", "day 6 + draw_count", "day 6 + draw_streak", "day 6 + lose_streak", "day 6 + loserate", "day 6 + winning_margin", "day 6 + winrate", "day 7 + day 8", "day 7 + day 9", "day 7 + draw_streak", "day 7 + lose_count", "day 7 + loserate", "day 7 + winlose", "day 8 + draw_streak", "day 8 + lose_count", "day 9 + lose_count", "draw_count + lose_count", "draw_count + loserate", "draw_count + win_streak", "draw_count + winrate", "draw_streak + drawrate", "draw_streak + lose_count", "draw_streak + lose_streak", "draw_streak + loserate", "draw_streak + win_count", "draw_streak + winlose", "draw_streak + winrate", "drawrate + lose_streak", "drawrate + win_streak", "drawrate + winlose", "drawrate + winning_margin", "drawrate + winrate", "lose_count + win_streak", "lose_count + winlose", "lose_count + winrate", "lose_streak + loserate", "lose_streak + win_count", "lose_streak + win_streak", "lose_streak + winrate", "loserate + win_count", "loserate + win_streak", "loserate + winlose", "loserate + winning_margin", "loserate + winrate", "win_count + win_streak", "win_count + winlose", "win_streak + winrate", "winlose + winning_margin", "winlose + winrate", "bet_amount * day 13", "bet_amount * lose_count", "bet_amount * lose_streak", "bet_amount * win_streak", "bet_amount * winning_margin", "bet_count * draw_streak", "bet_count * drawrate", "bet_count * loserate", "bet_count * win_streak", "bet_count * winlose", "bet_count * winning_margin", "bet_count * winrate", "day 1 * day 8", "day 1 * drawrate", "day 1 * lose_streak", "day 1 * winning_margin", "day 10 * day 11", "day 10 * day 13", "day 10 * day 2", "day 10 * day 6", "day 12 * day 3", "day 12 * day 6", "day 12 * day 7", "day 12 * day 8", "day 12 * lose_streak", "day 13 * day 6", "day 13 * day 7", "day 13 * day 8", "day 13 * lose_count", "day 13 * lose_streak", "day 13 * loserate", "day 13 * win_count", "day 13 * winlose", "day 13 * winning_margin", "day 13 * winrate", "day 2 * day 6", "day 2 * day 7", "day 2 * day 8", "day 2 * winlose", "day 3 * day 6", "day 3 * draw_count", "day 4 * day 5", "day 4 * day 6", "day 4 * draw_streak", "day 6 * day 7", "day 6 * day 8", "day 6 * drawrate", "day 7 * day 8", "day 7 * drawrate", "day 7 * lose_count", "day 7 * lose_streak", "day 7 * winlose", "day 7 * winning_margin", "day 7 * winrate", "day 8 * day 9", "day 8 * winlose", "draw_count * draw_streak", "draw_count * lose_count", "draw_count * win_streak", "draw_count * winning_margin", "draw_streak * drawrate", "draw_streak * lose_count", "draw_streak * lose_streak", "draw_streak * loserate", "draw_streak * win_streak", "draw_streak * winlose", "draw_streak * winning_margin", "drawrate * lose_count", "drawrate * lose_streak", "drawrate * loserate", "drawrate * win_count", "drawrate * winning_margin", "drawrate * winrate", "lose_count * win_streak", "lose_count * winlose", "lose_count * winning_margin", "lose_count * winrate", "lose_streak * loserate", "lose_streak * win_streak", "lose_streak * winlose", "lose_streak * winning_margin", "lose_streak * winrate", "loserate * win_streak", "loserate * winlose", "loserate * winning_margin", "loserate * winrate", "win_count * win_streak", "win_count * winlose", "win_count * winning_margin", "win_count * winrate", "win_streak * winlose", "win_streak * winrate", "winlose * winrate", "winning_margin * winrate"]

    subset_df = X[selected_columns]

    # Make predictions using your model
    prediction = load_model.predict(subset_df)

    # You can return the prediction as JSON response
    response = {'prediction': prediction.tolist()}
    return jsonify(response)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
