# Huấn luyện, lựa chọn theo CV, hiệu chỉnh khoảng dự báo và đánh giá ngoài mẫu.
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor

class PersistenceRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]

class ReturnForecaster(RegressorMixin, BaseEstimator):
    """Học lợi suất thay vì mức giá; scaler chỉ fit trên phần train của mỗi fold."""
    def __init__(self, estimator=None, shrinkage=1.0):
        self.estimator = estimator
        self.shrinkage = shrinkage

    @staticmethod
    def transform(X):
        z = np.asarray(X, dtype=float).copy()
        close = z[:, 0].copy()
        for i in (1, 4, 5):
            z[:, i] = z[:, i] / close - 1
        z[:, 0] = 0
        return z

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), np.asarray(y) / np.asarray(X)[:, 0] - 1)
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0] * (1 + self.shrinkage * self.estimator_.predict(self.transform(X)))

def metric_values(y, pred):
    return dict(MAE=float(mean_absolute_error(y, pred)),
                RMSE=float(np.sqrt(np.mean((np.asarray(y)-pred)**2))),
                R2=float(r2_score(y, pred)))

def select_model(candidates, X, y, cv, domain):
    searches, rows = {}, []
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(estimator, grid or [{}], scoring="neg_mean_absolute_error",
                              cv=cv, n_jobs=1, refit=True, error_score="raise")
        search.fit(X, y)
        searches[name] = search
        rows.append({"Miền": domain, "Mô hình": name,
                     "CV_MAE": float(-search.best_score_), "Tham số": str(search.best_params_)})
    name = min(searches, key=lambda k: -searches[k].best_score_)
    return name, searches[name], rows

def calibrate(model, X_cal, y_cal, X_test, y_test, relative=False):
    cal_pred = model.predict(X_cal)
    denom = np.asarray(X_cal)[:, 0] if relative else 1.0
    residual = np.abs(np.asarray(y_cal)-cal_pred) / denom
    level = min(1.0, np.ceil((len(residual)+1)*0.90)/len(residual))
    radius = float(np.quantile(residual, level, method="higher"))
    test_radius = radius * np.asarray(X_test)[:, 0] if relative else radius
    coverage = float(np.mean(np.abs(np.asarray(y_test)-model.predict(X_test)) <= test_radius))
    return {"nominal": 0.90, "radius": radius, "relative": relative,
            "calibration_size": len(y_cal), "test_coverage": coverage}

X, y = diabetes_df.drop(columns="target"), diabetes_df["target"]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, random_state=RANDOM_SEED)
X_train, X_cal, y_train, y_cal = train_test_split(X_dev, y_dev, test_size=.2, random_state=RANDOM_SEED)
diabetes_candidates = {
    "Median baseline": (DummyRegressor(strategy="median"), {}),
    "Linear": (Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]), {}),
    "Ridge": (Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
              {"model__alpha": [.01,.1,1,10,50,100]}),
    "ElasticNet": (Pipeline([("scale", StandardScaler()), ("model", ElasticNet(max_iter=10000))]),
                   {"model__alpha": [.05,.5,2], "model__l1_ratio": [.1,.5,.9]}),
    "Huber": (Pipeline([("scale", StandardScaler()), ("model", HuberRegressor(max_iter=2000))]),
              {"model__epsilon": [1.35,1.75,2.0]}),
    "Random Forest": (RandomForestRegressor(n_estimators=160, random_state=RANDOM_SEED, n_jobs=1),
                      {"min_samples_leaf": [4,8,12], "max_features": [.7,1.0]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_SEED),
                          {"n_estimators": [80,160], "learning_rate": [.025,.05], "max_depth": [1,2], "min_samples_leaf": [10]}),
}
diabetes_model_name, diabetes_search, diabetes_cv_rows = select_model(
    diabetes_candidates, X_train, y_train, KFold(5, shuffle=True, random_state=RANDOM_SEED), "Tiểu đường")
diabetes_model = diabetes_search.best_estimator_
diabetes_pred = diabetes_model.predict(X_test)
diabetes_interval = calibrate(diabetes_model, X_cal, y_cal, X_test, y_test)
diabetes_metrics = dict(Model=diabetes_model_name, **metric_values(y_test, diabetes_pred),
                        CV_MAE=float(-diabetes_search.best_score_), train_size=len(X_train),
                        calibration_size=len(X_cal), test_size=len(X_test),
                        Baseline_MAE=float(mean_absolute_error(y_test, np.repeat(y_train.median(),len(y_test)))))
best_params = diabetes_search.best_params_
perm = permutation_importance(diabetes_model, X_test, y_test, n_repeats=10,
                              random_state=RANDOM_SEED, scoring="neg_mean_absolute_error")
feature_importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

def market_summary(df, currency):
    clean = df.dropna(subset=["Close"])
    latest, first = clean.iloc[-1], clean.iloc[0]
    return {"start": clean.index.min().strftime("%d/%m/%Y"), "end": clean.index.max().strftime("%d/%m/%Y"),
            "observations": len(clean), "latest_close": float(latest.Close),
            "period_return_pct": float((latest.Close/first.Close-1)*100),
            "volatility_pct": float(clean.Return_1D.std()*np.sqrt(252)*100),
            "ma20": float(latest.MA20), "ma60": float(latest.MA60), "rsi14": float(latest.RSI14), "currency": currency}

def prepare_market_supervised(df):
    work = df.copy().sort_index()
    work["Volume_Change"] = work.Volume.replace(0,np.nan).pct_change(fill_method=None)
    work["Range_Pct"] = (work.High-work.Low)/work.Close
    work["Open_Close_Pct"] = (work.Close-work.Open)/work.Open
    work["Close_Lag1"] = work.Close.shift(1)
    work["Return_Lag1"] = work.Return_1D.shift(1)
    work["Target_Next_Close"] = work.Close.shift(-1)
    cols = ["Close","Close_Lag1","Return_1D","Return_Lag1","MA20","MA60","Volatility20","RSI14","Volume_Change","Range_Pct","Open_Close_Pct"]
    latest = work[cols].replace([np.inf,-np.inf],np.nan).dropna().iloc[-1]
    clean = work[cols+["Target_Next_Close"]].replace([np.inf,-np.inf],np.nan).dropna()
    return clean, cols, latest

market_details = {}
def train_market_model(df, domain_label):
    supervised, cols, latest = prepare_market_supervised(df)
    # Purge 1 quan sát tại mỗi ranh giới: nhãn t+1 không vượt vào block kế tiếp.
    a, b = int(len(supervised)*.64), int(len(supervised)*.80)
    train, cal, test = supervised.iloc[:a-1], supervised.iloc[a:b-1], supervised.iloc[b:]
    Xtr,ytr = train[cols],train.Target_Next_Close
    Xte,yte = test[cols],test.Target_Next_Close
    ridge = Pipeline([("scale",StandardScaler()),("model",Ridge())])
    candidates = {
        "Persistence": (PersistenceRegressor(), {}),
        "Ridge price": (ridge, {"model__alpha":[.01,1,10,100]}),
        "Ridge return": (ReturnForecaster(ridge), {"estimator__model__alpha":[1,10,100,1000],"shrinkage":[.1,.35,1.0]}),
        "Forest return": (ReturnForecaster(RandomForestRegressor(n_estimators=120,random_state=RANDOM_SEED,n_jobs=1)),
                          {"estimator__min_samples_leaf":[8,20],"shrinkage":[.1,.35]}),
        "Boost return": (ReturnForecaster(GradientBoostingRegressor(n_estimators=80,max_depth=1,min_samples_leaf=15,learning_rate=.03,random_state=RANDOM_SEED)),
                         {"shrinkage":[.1,.35,1.0]}),
    }
    name,search,rows = select_model(candidates,Xtr,ytr,TimeSeriesSplit(n_splits=5,gap=1),domain_label)
    model = search.best_estimator_
    pred,base = model.predict(Xte),Xte.Close.to_numpy()
    metrics = dict(Model=name, **metric_values(yte,pred), CV_MAE=float(-search.best_score_),
                   MAPE=float(np.mean(np.abs((np.asarray(yte)-pred)/np.asarray(yte)))*100),
                   Direction_Accuracy=float(np.mean(np.sign(np.asarray(yte)-base)==np.sign(pred-base))*100),
                   Direction_Coverage=float(np.mean(np.abs(pred-base)>1e-8)*100),
                   Persistence_MAE=float(mean_absolute_error(yte,base)),best_params=search.best_params_,
                   train_size=len(train),calibration_size=len(cal),test_size=len(test))
    metrics["Improvement_vs_Persistence_pct"]=(1-metrics["MAE"]/metrics["Persistence_MAE"])*100
    interval = calibrate(model,cal[cols],cal.Target_Next_Close,Xte,yte,relative=True)
    predictions = pd.DataFrame({"Actual":yte,"Prediction":pred,"Persistence":base},index=yte.index)
    market_details[domain_label] = {"interval":interval,"train":Xtr,"test":Xte,"rows":rows}
    return metrics,predictions,rows,model,cols,latest

stock_summary = market_summary(stock_df,stock_meta.get("currency","VND"))
gold_summary = market_summary(gold_df,gold_meta.get("currency","USD"))
stock_model_info,stock_predictions,stock_cv_rows,stock_model,market_feature_cols,stock_latest_features = train_market_model(stock_df,"Chứng khoán FPT")
gold_model_info,gold_predictions,gold_cv_rows,gold_model,gold_feature_cols,gold_latest_features = train_market_model(gold_df,"Giá vàng GC=F")
model_comparison_df = pd.DataFrame(diabetes_cv_rows+stock_cv_rows+gold_cv_rows)
model_comparison_df.to_csv(DATA_DIR/"model_cv_comparison.csv",index=False)
stock_predictions.to_csv(DATA_DIR/"fpt_test_predictions.csv")
gold_predictions.to_csv(DATA_DIR/"gold_test_predictions.csv")
data_profile_df = pd.DataFrame([
    {"Miền":"Tiểu đường","Nguồn":"scikit-learn load_diabetes","Số dòng":len(X),"Số cột":11,"Phạm vi":"442 mẫu nghiên cứu","Biến đích/đơn vị":"tiến triển sau 1 năm"},
    *[{"Miền":label,"Nguồn":market_sources[key],"Số dòng":len(df),"Số cột":len(df.columns),
       "Phạm vi":summary['start']+' - '+summary['end'],"Biến đích/đơn vị":"Close phiên kế tiếp"}
      for key,label,df,summary in [("stock","Chứng khoán",stock_df,stock_summary),("gold","Giá vàng",gold_df,gold_summary)]]])
summary_table = pd.DataFrame([{"Miền":label,**metrics} for label,metrics in
                              [("Tiểu đường",diabetes_metrics),("FPT",stock_model_info),("Vàng",gold_model_info)]])
display(summary_table[["Miền","Model","CV_MAE","MAE","R2"]].round(4))
print("Khoảng dự báo hiệu chỉnh riêng, mục tiêu 90%; coverage test được báo cáo thực tế.")
