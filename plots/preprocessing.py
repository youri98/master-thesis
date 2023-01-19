import pandas as pd
import os

def preprocess(location, column_name="Visited Rooms", x_axis="N Frames", steps = [idx for idx in range(0, int(1e9) + 1, int(1e5), )]):
    df = pd.read_csv(location)
    df = df.set_index(df[x_axis])
    # df = df.iloc[:, 1::3]
    df = df.filter(regex=(f"^.*{column_name}$"))

    idxs = steps
    
    df = df.append(pd.DataFrame([], index=idxs, columns=df.columns))
    df = df.sort_index()

    df = df.fillna(method='ffill')
    df = df.fillna(1)
    df = df.loc[idxs, :]

    for col in df.columns:
        df[col + " max"] = df[col].cummax()

    max_df = df.filter(regex=".* max").iloc[-1,:]

    results = []
    for col in df.filter(regex=".* max").columns:
        room = max_df[col]
        # print(col, df[col][df[col].eq(room)].index)
        results.append(df[col][df[col].eq(room)].index[0])


    max_df = df.filter(regex=".* max").iloc[-1,:].to_frame().reset_index().rename(columns={"index": "algo"})
    max_df = max_df.rename(columns={max_df.columns[1]: "room"})

    max_df["frame"] = pd.Series(results)
    max_df["algo"] = max_df["algo"].str[:-20]
    max_df = max_df.sort_values(by=["algo"]).reset_index(drop=True)

    df = df.filter(regex=(f"^.*{column_name} max$"))
    df = df.stack(dropna=True).reset_index()
    df = df.rename(columns={"level_1": "algo", 0:"room", "Step":"step"})
    df["algo"] = df["algo"].str[:-19]

    
    return max_df, df

preprocess("data/time/default_time.csv", x_axis="Relative Time (Process)", steps= [idx for idx in range(0, int(100) + 1)])