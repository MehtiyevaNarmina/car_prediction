import streamlit as st
import pandas as pd
import pickle
st.title('Car Price Prediction')
df_st = pd.read_csv('cleaned_data.csv')

new = st.sidebar.toggle(label = 'New')
if new == True:
    df_st = df_st.loc[df_st['New'] == 'Bəli']
    new = 'Bəli'
else:
    df_st = df_st.loc[df_st['New'] == 'Xeyr']
    new = 'Xeyr'

marks = st.sidebar.selectbox(
    label = 'Mark',
    options = list(df_st['Marka'].unique())
)
model = st.sidebar.selectbox(
    label = 'Model',
    options = list(df_st.loc[df_st['Marka'] == marks, 'Model'].unique())
)
year = st.sidebar.number_input(
    label = 'Year',
    min_value = df_st['Year'].min(),
    max_value = df_st['Year'].max(),
    value = df_st['Year'].max()
)

mileage = st.sidebar.number_input(
    label = 'Mileage',
    min_value = 0,
    max_value = df_st['Mileage'].max()
)
engine = st.sidebar.selectbox(
    label = 'Engine',
    options = list(df_st.loc[df_st['Model'] == model, 'Engine'].unique())

)
gear_box = st.sidebar.radio(
    label = 'Gear box',
    options = df_st.loc[df_st['Model'] == model, 'Gear_box'].unique()
)
fuel_type = st.sidebar.radio(
    label = 'Fuel type',
    options = df_st.loc[df_st['Model'] == model, 'Fuel_type'].unique()
)
color = st.sidebar.selectbox(
    label = 'Color',
    options = list(df_st['Color'].unique())
)

# filters = pd.DataFrame(dat = [marks, model, year, color, engine, fuel_type, mileage, gear_box, new ], columns = df_st.drop(['Price'], axis = 1).columns)
filters = pd.DataFrame(data = {'Year': year, 'Color': color, 'Engine': engine,
                               'Fuel_type':fuel_type, 'Mileage': mileage,
                               'Gear_box': gear_box, 'New': new,
                               'Marka': marks, 'Model': model}, index = [0])
# st.dataframe(filters)

with open("color.pkl", "rb") as f:
        color = pickle.load(f)

filters['Color'] = color.transform(filters['Color'])

with open("fuel_type.pkl", "rb") as f:
        fuel_type = pickle.load(f)

filters['Fuel_type'] = fuel_type.transform(filters['Fuel_type'])

with open("gear_box.pkl", "rb") as f:
        gear_box = pickle.load(f)

filters['Gear_box'] = gear_box.transform(filters['Gear_box'])

with open("new.pkl", "rb") as f:
        new = pickle.load(f)

filters['New'] = new.transform(filters['New'])

with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

encoder_df = encoder.transform(filters.select_dtypes('object'))

encoder_df = pd.DataFrame(encoder_df, columns = encoder.get_feature_names_out(), index=filters.index)

encoder_df = pd.concat([filters.drop(['Marka','Model'], axis = 1).reset_index(drop = True), encoder_df.reset_index(drop = True)], axis = 1)

with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
predicted_price = model.predict(encoder_df)
st.subheader(f'Predicted Price: {int(predicted_price[0])}')