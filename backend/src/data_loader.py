import fastf1
import pandas as pd


def load_data(filepath="data/qualifying_results.csv"):
    return pd.read_csv(filepath)


def fetch_qualifying_results(year=2024):
    results = {}
    schedule = fastf1.get_event_schedule(year)
    for race in schedule['EventName'].unique(): 
        session = fastf1.get_session(year, race, 'Q')
        session.load()
        results[race] = session.results[['Abbreviation', 'TeamName', 'Q1', 'Q2', 'Q3', 'Position']]
    return results

def compile_results_to_df(results):
    for race, df in results.items():
        df['Race'] = race
    combined_df = pd.concat(results.values())
    return combined_df