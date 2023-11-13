import streamlit as st
import plots as p
import joblib
import pandas as pd


st.set_page_config(page_title="Machine Learning Model", layout="wide", page_icon='🤖')

st.markdown("<h1 style='text-align: center;'>🤖 Machine Learning Model</h1>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""##### ⚖️ Data Splitting""", unsafe_allow_html=True)

st.code("""

# Dropping columns that are not required for modeling casual + registered = cnt
X = df.drop(columns=['cnt', 'dteday', 'casual', 'registered'])

# Target variable
y = df['cnt']
# Splitting the data into training (60%), validation (20%), and testing (20%) sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

X_train.shape, X_val.shape, X_test.shape
        
""", language='python')

st.markdown("---")

st.markdown("""### 1️⃣ Linear Regression""", unsafe_allow_html=True)

st.markdown("###### Evaluation Metrics")
mae_values = {'train': 8.2300, 'validation': 7.8869, 'test': 8.0224}
mse_values = {'train': 428.9310, 'validation': 409.0490, 'test': 409.4724}
r2_values = {'train': 0.1718, 'validation': 0.1824, 'test': 0.1597}

table_data = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Train': [mae_values['train'], mse_values['train'], r2_values['train']],
    'Validation': [mae_values['validation'], mse_values['validation'], r2_values['validation']],
    'Test': [mae_values['test'], mse_values['test'], r2_values['test']]
}
st.dataframe(table_data)

y_test = joblib.load('./joblib_files/y_test.plk')
pred_test_linear = joblib.load('./joblib_files/pred_linear_reg.plk')
grid_rf = joblib.load('joblib_files/grid_search_RF.plk')

st.plotly_chart(p.plot_predictions(500, y_test, 'Linear Regression', pred_test_linear))

st.markdown("---")

st.markdown("""### 2️⃣ Random Forest""", unsafe_allow_html=True)


# Assuming you have the MAE, MSE, and R2 values as a DataFrame
data_metrics = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Train': [0.9880, 36.4599, 0.9296],
    'Validation': [1.7506, 124.4141, 0.7513],
    'Test': [1.6587, 112.3184, 0.7695]
}
df_metrics = pd.DataFrame(data_metrics)

# Assuming you have another DataFrame with the hyperparameters
data_hyperparameters = {
    'Parameter': ['max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators'],
    'Value': [None, 2, 5, 200]
}
df_hyperparameters = pd.DataFrame(data_hyperparameters)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("###### Evaluation Metrics")
    st.dataframe(df_metrics.style.set_properties(**{'text-align': 'center'}), hide_index=True)

with col2:
    st.markdown("###### Best Hyperparameters")
    st.dataframe(df_hyperparameters.style.format({'Value': '{:.0f}'}).set_properties(**{'text-align': 'center'}), hide_index=True)


pred_test_rf = joblib.load('./joblib_files/pred_random_forest.plk')

st.plotly_chart(p.plot_predictions(500, y_test, 'Random Forest Regressor', pred_test_rf))
st.markdown("---")


st.markdown("""### 3️⃣ XGBoost""", unsafe_allow_html=True)
    
data_metrics = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Train': [2.0119, 41.878, 0.9191],
    'Validation': [3.3658, 139.1574, 0.7219],
    'Test': [3.1496, 138.7585, 0.7152]
}
df_metrics = pd.DataFrame(data_metrics)

# Assuming you have another DataFrame with the hyperparameters
data_hyperparameters = {
    'Parameter': ['colsample_bytree', 'learning_rate', 'max_depth', 'min_child_weight', 'n_estimators', 'subsample'],
    'Value': [0.4, 0.05, 12, 15, 500, 0.8]
}

df_hyperparameters = pd.DataFrame(data_hyperparameters)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("###### Evaluation Metrics")
    st.dataframe(df_metrics.style.set_properties(**{'text-align': 'center'}), hide_index=True)

with col2:
    st.markdown("###### Best Hyperparameters")
    st.dataframe(df_hyperparameters.style.format({'Value': '{:.2f}'}).set_properties(**{'text-align': 'center'}), hide_index=True)


pred_test_xgb = joblib.load('./joblib_files/pred_xgb.plk')

st.plotly_chart(p.plot_predictions(500, y_test, 'XGBoost Regressor', pred_test_xgb))
st.markdown("---")

st.markdown("""### 4️⃣ LightGBM""", unsafe_allow_html=True)
    
data_metrics = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Train': [1.8327, 36.324, 0.9299],
    'Validation': [2.9053, 119.3318, 0.7615],
    'Test': [2.6601, 117.8442, 0.7582]
}
df_metrics = pd.DataFrame(data_metrics)

# Assuming you have another DataFrame with the hyperparameters
data_hyperparameters = {
    'Parameter': ['colsample_bytree', 'early_stopping_rounds', 'learning_rate', 'n_estimators', 'subsample'],
    'Value': [0.5, 100, 0.05, 2000, 0.7]
}

df_hyperparameters = pd.DataFrame(data_hyperparameters)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("###### Evaluation Metrics")
    st.dataframe(df_metrics.style.set_properties(**{'text-align': 'center'}), hide_index=True)

with col2:
    st.markdown("###### Best Hyperparameters")
    st.dataframe(df_hyperparameters.style.format({'Value': '{:.2f}'}).set_properties(**{'text-align': 'center'}), hide_index=True)


pred_test_lbm = joblib.load('./joblib_files/pred_lightgbm.plk')

st.plotly_chart(p.plot_predictions(500, y_test, 'LightGBM Regressor', pred_test_lbm))
st.markdown("---")

st.markdown("""### 🌟 Final Model""")
            

st.markdown(""" **Feature Importance** """, unsafe_allow_html=True)
st.code("""

# Extract feature importance
feature_importances = final_model.feature_importance(importance_type='split')

# Create DataFrame for variable importance
var_imp = pd.DataFrame({'var': X_train.columns, 'imp': feature_importances})
var_imp.sort_values(['imp'], ascending=False, inplace=True)
        
# Select top 10 variables
top_var = var_imp.nlargest(33, 'imp')['var'].tolist()
train_data = lgb.Dataset(X_train[top_var], label=y_train)
valid_data = lgb.Dataset(X_val[top_var], label=y_val, reference=train_data)
final_model = lgb.train(best_params,
                  train_data, valid_sets=[train_data, valid_data])
        
""", language='python') 

st.success(""" **We have applied feature selection and the model has slightly improved using 33 features instead of 69.** """)

data_metrics = {
    'Metric': ['MAE', 'MSE', 'R2'],
    'Train': [1.8979, 46.1349, 0.9109],
    'Validation': [2.8076, 119.2248, 0.7617],
    'Test': [2.5197, 111.4118, 0.7714]
}
df_metrics = pd.DataFrame(data_metrics)

# Assuming you have another DataFrame with the hyperparameters
data_hyperparameters = {
    'Parameter': ['colsample_bytree', 'early_stopping_rounds', 'learning_rate', 'n_estimators', 'subsample'],
    'Value': [0.5, 100, 0.05, 2000, 0.7]
}

df_hyperparameters = pd.DataFrame(data_hyperparameters)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("###### Evaluation Metrics")
    st.dataframe(df_metrics.style.set_properties(**{'text-align': 'center'}), hide_index=True)

with col2:
    st.markdown("###### Best Hyperparameters")
    st.dataframe(df_hyperparameters.style.format({'Value': '{:.2f}'}).set_properties(**{'text-align': 'center'}), hide_index=True)

final_model = joblib.load('./joblib_files/final_model.plk')

st.plotly_chart(p.plot_predictions(500, y_test, 'Final Model', final_model))
st.markdown("---")

st.markdown("""### 🚀 Conclusions   
            
##### **1. Data Preprocessing** 
- The dataset underwent thorough preprocessing, addressing missing values, outliers, and encoding categorical variables. This set a strong foundation for the next steps.

##### **2. Model Exploration** 
- A diverse set of models, including Random Forest, XGBoost, Linear Regression, and LightGBM, were explored. They were evaluated using metrics such as MAE, MSE, and R2.

##### **3. Overfitting Challenge** 
- A common trend was overfitting, where models excelled with training data but faltered on validation/test sets. This hinted at models potentially being too intricate.

##### **4. The Role of Regularization** 
- To tackle overfitting, regularization came to the rescue, especially with tree-based models. It effectively penalizes models that align too closely with training data.

##### **5. The Best Performer - LightGBM** 
- Among all, LightGBM was the star. Its speed, capability to manage categories directly, and resistance to overfitting (with the right tuning) made it shine brighter than the rest.

---

### **Final Thoughts**
- LightGBM outperformed others in our tests. Thanks to careful regularization and feature selection, its efficiency was even more evident. This model's predictions will aid in better bike allocation, leading to cost savings and happier users. Plus, it ensures bikes are available as needed, cutting down on costs and amplifying user satisfaction.
           
               
            """, unsafe_allow_html=True)



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
