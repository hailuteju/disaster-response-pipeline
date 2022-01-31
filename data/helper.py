import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

root_path = Path(os.path.dirname(__file__)).parent.absolute()
db_path = f"{root_path}/data/DisasterResponse.db"
engine = create_engine(f'sqlite:///{db_path}')
messages_df = pd.read_sql_table('messages', engine)

messages_df.to_csv("data/messages.csv", index=False)