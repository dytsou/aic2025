# train.py
import os
import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR, DATA_SAVE_DIR

INDICATORS = ["boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma"]

def train_ppo_model(data_path, train_start_date=None, train_end_date=None, model_name='ppo', 
                    total_timesteps=80000, ppo_params=None, save_suffix='1D'):
    """
    Train a PPO model for stock trading using FinRL.
    
    Parameters:
    - data_path (str): Path to the CSV file containing stock data.
    - train_start_date (str, optional): Start date for training data. Defaults to min date in data.
    - train_end_date (str, optional): End date for training data. Defaults to max date in data.
    - model_name (str): RL algorithm name (default: 'ppo').
    - total_timesteps (int): Number of timesteps to train the model.
    - ppo_params (dict, optional): PPO hyperparameters. Defaults to predefined values.
    - save_suffix (str): Suffix for saved model filename.
    
    Returns:
    - trained_model: Trained PPO model object.
    """
    # Load data
    processed_full = pd.read_csv(data_path)
    train_start_date = train_start_date or processed_full['date'].min()
    train_end_date = train_end_date or processed_full['date'].max()
    train = data_split(processed_full, train_start_date, train_end_date)
    
    # Environment configs
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    # PPO configs
    default_ppo_params = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "batch_size": 128,
    }
    ppo_params = ppo_params or default_ppo_params
    
    # Ensure directories exist
    check_and_make_directories([TRAINED_MODEL_DIR])
    
    # Environment
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(f"Training environment type: {type(env_train)}")
    
    # PPO agent
    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model(model_name, model_kwargs=ppo_params)
    
    # Set up logger
    tmp_path = os.path.join(RESULTS_DIR, model_name)
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)
    
    # Train the model
    trained_ppo = agent.train_model(model=model_ppo,
                                   tb_log_name=model_name,
                                   total_timesteps=total_timesteps)
    
    # Save the model
    save_path = os.path.join(TRAINED_MODEL_DIR, f'trained_{model_name}_{save_suffix}')
    trained_ppo.save(save_path)
    print(f"Model saved to {save_path}")
    
    return trained_ppo

if __name__ == '__main__':
    data_path = os.path.join(DATA_SAVE_DIR, 'trade_data_1D.csv')
    trained_model = train_ppo_model(data_path=data_path)