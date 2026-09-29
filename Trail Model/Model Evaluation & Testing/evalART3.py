import sys,os,sqlite3,datetime,json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from ModelV2.ART4.ART import AdaptiveModel,run_art3_pipeline,IF_CONTAMINATION, Z_THRESHOLD, ROLLING_WINDOW, REFIT_N, RETRAIN_WINDOW

DB_PATH=os.path.join(os.path.dirname(__file__),'experiment_log.db')
XLSX_PATH=os.path.join(os.path.dirname(__file__),'experiment_log.xlsx')
MODEL_NAME='ART3'
CHUNK_SIZE=REFIT_N

def inject_spikes(df,n,rng):
    out=df.copy()
    available=out.index[out["is_anomaly"]==0]
    chosen=rng.choice(available,size=min(n,len(available)),replace=False)
    for i in chosen:
        out.loc[i,"temperature_c"]+=10*rng.choice([-1,1])
        out.loc[i,["is_anomaly","anomaly_type"]]=[1,"spike"]
    return out

def inject_frozen(df,n,rng):
    out=df.copy()
    length=12
    made=0
    tries=0
    while made<n and tries<n*200:
        tries+=1
        start=int(rng.integers(0,max(1,len(out)-length+1)))
        idx=list(range(start,start+length))
        if idx[-1]>=len(out):
            continue
        if (out.loc[idx,"is_anomaly"]==0).all():
            for f in ["temperature_c","humidity_pct","pressure_hpa"]:
                out.loc[idx,f]=out.loc[start,f]
            out.loc[idx,"is_anomaly"]=1
            out.loc[idx,"anomaly_type"]="frozen"
            made+=1
    return out

def inject_drift(df,n,rng):
    out=df.copy()
    length=30
    made=0
    tries=0
    while made<n and tries<n*200:
        tries+=1
        start=int(rng.integers(0,max(1,len(out)-length+1)))
        idx=list(range(start,start+length))
        if idx[-1]>=len(out):
            continue
        if (out.loc[idx,"is_anomaly"]==0).all():
            direction=rng.choice([-1,1])
            offset=np.linspace(0,8*direction,length)
            out.loc[idx,"temperature_c"]=out.loc[idx,"temperature_c"].to_numpy()+offset
            out.loc[idx,"is_anomaly"]=1
            out.loc[idx,"anomaly_type"]="drift"
            made+=1
    return out

def inject_dropout(df,n,rng):
    out=df.copy()
    available=out.index[out["is_anomaly"]==0]
    chosen=rng.choice(available,size=min(n,len(available)),replace=False)
    for i in chosen:
        if rng.random()<0.5:
            out.loc[i,"humidity_pct"]=rng.uniform(0,5)
        else:
            out.loc[i,"humidity_pct"]=rng.uniform(100,130)
        out.loc[i,["is_anomaly","anomaly_type"]]=[1,"dropout"]
    return out

def choose_anomalies():
    print("\nSelect anomaly type(s) to inject:")
    print("0 = None")
    print("1 = Spike")
    print("2 = Frozen value")
    print("3 = Gradual drift")
    print("4 = Dropout / noise")
    print("5 = All four")
    print("Example: 1,3 -> Spike + Gradual drift")
    print("         2,4 -> Frozen + Dropout")
    print("         0   -> Clean test data")

    while True:
        raw=input("\nEnter option number(s): ").strip()

        if raw=="0":
            return {}

        if raw=="5":
            selected=["spike","frozen","drift","dropout"]
            break

        try:
            nums=[int(x.strip()) for x in raw.split(",")]
            if len(nums)==len(set(nums)) and nums and all(x in [1,2,3,4] for x in nums):
                names={1:"spike",2:"frozen",3:"drift",4:"dropout"}
                selected=[names[x] for x in nums]
                break
        except ValueError:
            pass

        print("Invalid choice. Use 0, 1, 2, 3, 4, 5 or combinations such as 1,3.")

    counts={}
    for name in selected:
        while True:
            try:
                value=int(input(f"How many '{name}' anomalies/windows? "))
                if value>=0:
                    counts[name]=value
                    break
            except ValueError:
                pass
            print("Enter a whole number >= 0.")

    return counts

def inject_anomalies(df,counts,rng):
    out=df.copy().reset_index(drop=True)
    out["is_anomaly"]=0
    out["anomaly_type"]="normal"

    if "spike" in counts and counts["spike"]>0:
        out=inject_spikes(out,counts["spike"],rng)

    if "frozen" in counts and counts["frozen"]>0:
        out=inject_frozen(out,counts["frozen"],rng)

    if "drift" in counts and counts["drift"]>0:
        out=inject_drift(out,counts["drift"],rng)

    if "dropout" in counts and counts["dropout"]>0:
        out=inject_dropout(out,counts["dropout"],rng)

    return out

def evaluate(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    tn,fp,fn,tp=cm.ravel()
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,zero_division=0)
    recall=recall_score(y_true,y_pred,zero_division=0)
    f1=f1_score(y_true,y_pred,zero_division=0)

    print("\nConfusion matrix (rows=actual, cols=predicted) [0,1]:")
    print(cm)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "tn":tn,
        "fp":fp,
        "fn":fn,
        "tp":tp
    }

def setup_database():
    conn=sqlite3.connect(DB_PATH)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs(
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp TEXT,
        model_name TEXT,
        contamination REAL,
        z_threshold REAL,
        rolling_window INTEGER,
        refit_n INTEGER,
        retrain_window INTEGER,
        anomaly_config TEXT,
        n_injected INTEGER,
        n_flagged INTEGER,
        caught INTEGER,
        catch_pct REAL,
        accuracy REAL,
        precision_score REAL,
        recall_score REAL,
        f1_score REAL,
        true_negatives INTEGER,
        false_positives INTEGER,
        false_negatives INTEGER,
        true_positives INTEGER,
        training_rows INTEGER,
        total_refits INTEGER
    )
    """)

    conn.commit()
    return conn

def save_results(metrics,counts,n_injected,n_flagged,caught,catch_pct,adaptive):
    conn=setup_database()

    conn.execute("""
    INSERT INTO runs(
        run_timestamp,model_name,contamination,z_threshold,rolling_window,
        refit_n,retrain_window,anomaly_config,n_injected,n_flagged,caught,
        catch_pct,accuracy,precision_score,recall_score,f1_score,
        true_negatives,false_positives,false_negatives,true_positives,
        training_rows,total_refits
    )
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        datetime.datetime.now().isoformat(timespec="seconds"),
        MODEL_NAME,
        IF_CONTAMINATION,
        Z_THRESHOLD,
        ROLLING_WINDOW,
        REFIT_N,
        RETRAIN_WINDOW,
        json.dumps(counts),
        n_injected,
        n_flagged,
        caught,
        catch_pct,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["tn"],
        metrics["fp"],
        metrics["fn"],
        metrics["tp"],
        adaptive.total_training_rows,
        adaptive.total_refits
    ))

    conn.commit()

    try:
        pd.read_sql_query("SELECT * FROM runs ORDER BY run_id",conn).to_excel(
            XLSX_PATH,index=False,sheet_name="experiment_log"
        )
    except Exception as e:
        print(f"Excel export skipped: {e}")

    conn.close()

def main():
    csv_path=os.path.join(os.path.dirname(__file__),"AWS_Weather_5000_Clean.csv")
    df=pd.read_csv(csv_path)

    if len(df)<5000:
        raise ValueError("AWS_Weather_5000_Clean.csv must contain at least 5000 rows.")

    df=df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")

    train=df.iloc[:2000].copy().reset_index(drop=True)
    raw_test=df.iloc[2000:5000].copy().reset_index(drop=True)

    print("\nART3 Adaptive Model Evaluation")
    print("2000 clean rows = initial training")
    print("3000 rows = test/stream")
    print("Normal test rows are used for adaptive learning.")
    print("Injected anomaly rows are NOT used for adaptive learning.")

    counts=choose_anomalies()
    rng=np.random.default_rng()

    test=inject_anomalies(raw_test,counts,rng)
    n_injected=int(test["is_anomaly"].sum())

    print(f"\nInjected {n_injected} anomalous rows into {len(test)} test rows.")

    if n_injected>0:
        print("Breakdown:",test.loc[test["is_anomaly"]==1,"anomaly_type"].value_counts().to_dict())
    else:
        print("No anomalies injected - running clean test.")

    adaptive=AdaptiveModel(train,load_saved=True)

    all_results=[]

    for start in range(0,len(test),CHUNK_SIZE):
        chunk=test.iloc[start:start+CHUNK_SIZE].copy().reset_index(drop=True)

        chunk_results=run_art3_pipeline(
            chunk,
            adaptive.get_model()
        )

        all_results.append(chunk_results)

        for _,row in chunk.iterrows():
            row_dict=row.to_dict()

            if int(row_dict["is_anomaly"])==0:
                adaptive.addnrefit(row_dict,allow_training=True)

    adaptive.save_state()

    results=pd.concat(all_results,ignore_index=True)

    y_true=test["is_anomaly"].astype(int).to_numpy()
    y_pred=results["is_anomaly"].astype(int).to_numpy()

    caught=int(np.sum((y_true==1)&(y_pred==1)))
    flagged=int(y_pred.sum())
    catch_pct=(caught/n_injected*100) if n_injected>0 else 0.0

    if n_injected>0:
        print(f"\nCaught {caught} out of {n_injected} injected anomalies ({catch_pct:.1f}%)")

    print(f"Flagged {flagged} rows as anomalies out of {len(y_pred)} total rows.")

    metrics=evaluate(y_true,y_pred)

    print(f"\nART3 training rows accumulated: {adaptive.total_training_rows}")
    print(f"ART3 refits completed: {adaptive.total_refits}")

    print("\nDetected anomaly classifications:")
    detected=results[results["is_anomaly"]==True]

    if len(detected)>0:
        print(detected["anomaly_type"].value_counts().head(20).to_string())
    else:
        print("No anomalies detected.")

    save_results(
        metrics,
        counts,
        n_injected,
        flagged,
        caught,
        catch_pct,
        adaptive
    )

    print("\nRun saved to experiment_log.db and experiment_log.xlsx.")

if __name__=="__main__":
    main()
