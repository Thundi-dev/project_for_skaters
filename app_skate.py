import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных
df_grouped = pd.read_csv('df_grouped.csv')
scores = pd.read_csv('scores.csv')

def remove_zero_columns(df):
  cols_to_drop = []
  for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
      if (df[col] == 0).all():
        cols_to_drop.append(col)

  df = df.drop(columns=cols_to_drop)
  return df

df_grouped = remove_zero_columns(df_grouped)
# Копия данных для обработки
df_grouped2 = df_grouped.copy()

# Функция для удаления юнитов с малым количеством данных
def remove_units(df):
    unit_counts = df['unit_id'].value_counts()
    units_to_remove = unit_counts[unit_counts <= 3].index
    mask = ~df['unit_id'].isin(units_to_remove)
    return df[mask].sort_values(by='date_start')

df_grouped2 = remove_units(df_grouped2)

# Функция для обучения модели и предсказания
def model__(df_grouped2, unit, n):
    df_grouped2 = df_grouped2.query('unit_id == @unit').sort_values('date_start')
    train_data = df_grouped2.iloc[:-n]
    last_performances = df_grouped2.iloc[-n:]

    cols_to_check = [col for col in df_grouped2.columns if col.endswith('_element_perfect')]
    y_train = train_data[cols_to_check]
    X_train = train_data.drop(cols_to_check + ['date_start', 'date_end'], axis=1)

    y_test = last_performances[cols_to_check]
    X_test = last_performances.drop(cols_to_check + ['date_start', 'date_end'], axis=1)

    cols_to_keep = [col for col in y_train.columns if y_train[col].nunique() >= 2]
    y_train = y_train[cols_to_keep]
    y_test = y_test[cols_to_keep]

    scaler = MinMaxScaler()
    num_columns = X_train.select_dtypes(['int', 'float']).columns.tolist()
    X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
    X_test[num_columns] = scaler.transform(X_test[num_columns])

    ohe_columns = X_train.select_dtypes('object').columns.tolist()
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[ohe_columns])
    ohe_train = ohe.transform(X_train[ohe_columns])
    ohe_test = ohe.transform(X_test[ohe_columns])
    ohe_feature_names = ohe.get_feature_names_out(ohe_columns)
    X_train_ohe = pd.DataFrame(ohe_train, index=X_train.index, columns=ohe_feature_names)
    X_test_ohe = pd.DataFrame(ohe_test, index=X_test.index, columns=ohe_feature_names)
    X_train = pd.concat([X_train.drop(ohe_columns, axis=1), X_train_ohe], axis=1)
    X_test = pd.concat([X_test.drop(ohe_columns, axis=1), X_test_ohe], axis=1)

    model = MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=42))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(predictions, index=X_test.index, columns=y_test.columns)

    probabilities = []
    for estimator, col in zip(model.estimators_, y_train.columns):
        proba = estimator.predict_proba(X_test)
        probabilities.append(pd.DataFrame(proba[:, 1], index=X_test.index, columns=[col]))
    probabilities_df = pd.concat(probabilities, axis=1)

    binary_predictions = (probabilities_df >= 0.25).astype(int)

    real = sorted(set([col.replace('_element_perfect', '') for col in y_test.columns if y_test[col].any()]))
    binary = sorted(set([col.replace('_element_perfect', '') for col in binary_predictions.columns if binary_predictions[col].any()]))

    return real, binary
elements_col = [col for col in df_grouped.columns if col.endswith('_element_perfect')]
elements = df_grouped.groupby('unit_id')[elements_col].sum().map(lambda x: 1 if x != 0 else 0)

new_columns = [col.replace('_element_perfect', '') if col.endswith('_element_perfect') else col for col in elements.columns]
elements.columns = new_columns
# Функция для рекомендаций
def recommendation(df, unit, n):
    # Вычисляем матрицу косинусной схожести
    similarity_matrix = cosine_similarity(df)
    indx = df.index.get_loc(unit)
    unit_elements = df.iloc[indx]
    best_elements = unit_elements[unit_elements > 0].index.tolist()
    similar_units = np.argsort(-similarity_matrix[indx])[1:]  # argsort возвращает индексы
    new_elements = []
    for unit_idx in similar_units:
        similar_unit_elements = df.iloc[unit_idx]
        new_elements.extend(similar_unit_elements[similar_unit_elements > 0].index.tolist())
    unique_new_elements = list(set(new_elements))  # Убираем дубликаты
    recommended_elements = [el for el in unique_new_elements if el not in best_elements][:n]
    similar_unit_ids = df.index[similar_units].tolist()[:n]
    return similar_unit_ids, recommended_elements

# Функция для поиска лучшего элемента
def best_el(unit):
    elements_reset = elements[elements.index == unit].reset_index()
    melted_df = elements_reset.melt(id_vars=['unit_id'], var_name='elements', value_name='value')
    filtered_df = melted_df[melted_df['value'] == 1].drop(columns=['value']).sort_values('unit_id')
    unit_score = filtered_df.merge(scores[['element', 'base']], left_on='elements', right_on='element').drop('elements', axis=1)
    max_id = unit_score.groupby('unit_id')['base'].idxmax()
    return unit_score.iloc[max_id]

# красивая таблица для дополнительной информации
# Создаем DataFrame с описанием элементов
data = {
    "Элемент": [
        "2A", "3A", "2F", "3F", "4F", "2Lo", "3Lo", "4Lo", "2Lz", "3Lz",
        "4Lz", "2S", "3S", "4S", "2T", "3T", "4T", "CCSp2", "CCSp3", "CCSp4",
        "CCoSp2", "CCoSp3", "CCoSp4", "CCoSpB", "CSSp2", "CSSp3", "CSSp4", "CSSpB",
        "CSp2", "CSp3", "CSp4", "CSpB", "FCCSp4", "FCCoSp2", "FCCoSp3", "FCCoSp4",
        "FCCoSpB", "FCSSp2", "FCSSp3", "FCSSp4", "FCSSpB", "FCSp2", "FCSp3", "FCSp4",
        "FCSpB", "FLSp2", "FLSp3", "FLSp4", "FSSp2", "FSSp3", "FSSp4", "FSSpB",
        "FSsp2", "LSp2", "LSp3", "LSp4", "SSp2", "SSp3", "SSp4", "SSpB",
        "USpB", "StSq2", "StSq3", "StSq4", "StSqB",
        "A", "CCSpB", "CCSp", "CCoSp", "CSSp", "CSp", "ChS", "ChSpl", "CoSp2", "CoSpB",
        "CoSp", "Eu", "FCCoSp", "FCSSp", "FCSp", "FCoSp2", "FCoSp", "FSSp", "FUSp", "F",
        "LSpB", "LSp", "Lo", "Lz", "SSp", "S", "Sp", "StSq", "T", "W"
    ],
    "Тип элемента": [
        "Прыжок", "Прыжок", "Прыжок", "Прыжок", "Прыжок",
        "Прыжок", "Прыжок", "Прыжок", "Прыжок", "Прыжок",
        "Прыжок", "Прыжок", "Прыжок", "Прыжок", "Прыжок",
        "Прыжок", "Прыжок", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Дорожка шагов",
        "Дорожка шагов", "Дорожка шагов", "Дорожка шагов",
        "Прыжок", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение",
        "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Вращение", "Прыжок",
        "Вращение", "Вращение", "Прыжок", "Прыжок", "Вращение", "Прыжок", "Вращение", "Дорожка шагов", "Прыжок", "Вращение"
    ],
    "Описание": [
        "Двойной аксель", "Тройной аксель", "Двойной флип", "Тройной флип", "Четверной флип",
        "Двойной луп", "Тройной луп", "Четверной луп", "Двойной лутц", "Тройной лутц",
        "Четверной лутц", "Двойной сальхов", "Тройной сальхов", "Четверной сальхов", "Двойной тулуп",
        "Тройной тулуп", "Четверной тулуп", "Каскадное вращение 2 уровня", "Каскадное вращение 3 уровня", "Каскадное вращение 4 уровня",
        "Комбинированное вращение 2 уровня", "Комбинированное вращение 3 уровня", "Комбинированное вращение 4 уровня", "Комбинированное вращение базового уровня",
        "Сит-спин 2 уровня", "Сит-спин 3 уровня", "Сит-спин 4 уровня", "Сит-спин базового уровня",
        "Вращение в ласточке 2 уровня", "Вращение в ласточке 3 уровня", "Вращение в ласточке 4 уровня", "Вращение в ласточке базового уровня",
        "Летящее каскадное вращение 4 уровня", "Летящее комбинированное вращение 2 уровня", "Летящее комбинированное вращение 3 уровня", "Летящее комбинированное вращение 4 уровня",
        "Летящее комбинированное вращение базового уровня", "Летящий сит-спин 2 уровня", "Летящий сит-спин 3 уровня", "Летящий сит-спин 4 уровня",
        "Летящий сит-спин базового уровня", "Летящее вращение в ласточке 2 уровня", "Летящее вращение в ласточке 3 уровня", "Летящее вращение в ласточке 4 уровня",
        "Летящее вращение в ласточке базового уровня", "Летящее вращение ласточкой 2 уровня", "Летящее вращение ласточкой 3 уровня", "Летящее вращение ласточкой 4 уровня",
        "Летящий сит-спин 2 уровня", "Летящий сит-спин 3 уровня", "Летящий сит-спин 4 уровня", "Летящий сит-спин базового уровня",
        "Летящий сит-спин 2 уровня", "Вращение ласточкой 2 уровня", "Вращение ласточкой 3 уровня", "Вращение ласточкой 4 уровня",
        "Сит-спин 2 уровня", "Сит-спин 3 уровня", "Сит-спин 4 уровня", "Сит-спин базового уровня",
        "Вращение вверх базового уровня", "Дорожка шагов 2 уровня", "Дорожка шагов 3 уровня", "Дорожка шагов 4 уровня",
        "Дорожка шагов базового уровня",
        "Аксель", "Каскадное вращение базового уровня", "Каскадное вращение", "Комбинированное вращение", "Сит-спин",
        "Вращение в ласточке", "Вращение в крюке", "Вращение в крюке с переходом", "Комбинированное вращение 2 уровня", "Комбинированное вращение базового уровня",
        "Комбинированное вращение", "Вращение в ласточке с переходом", "Летящее комбинированное вращение", "Летящий сит-спин", "Летящее вращение в ласточке",
        "Летящее комбинированное вращение 2 уровня", "Летящее комбинированное вращение", "Летящий сит-спин", "Летящее вращение вверх", "Флип",
        "Вращение в ласточке базового уровня", "Вращение в ласточке", "Луп", "Лутц", "Сит-спин", "Сальхов", "Вращение", "Дорожка шагов", "Тулуп", "Вращение в ласточке с переходом"
    ]
}

# Создаем DataFrame
df_elements = pd.DataFrame(data)

# Добавляем стили для красивого отображения
def color_type(val):
    if val == "Прыжок":
        return "background-color: #FFCCCC;color: #000000;"  # Светло-красный
    elif val == "Вращение":
        return "background-color: #CCE5FF;color: #000000;"  # Светло-голубой
    elif val == "Дорожка шагов":
        return "background-color: #CCFFCC;color: #000000;"  # Светло-зеленый
    return ""

# Применяем стили
styled_df = df_elements.style.applymap(color_type, subset=["Тип элемента"])


# Интерфейс Streamlit
# Настройка страницы на всю ширину
st.set_page_config(layout="wide")

image = Image.open("wallpaper_0201340-optimized.jpg")
st.image(image)

st.markdown(
    """
    <h1 style='text-align: center;'>Рекомендательная система для спортсменов</h1>
    <p style='text-align: center;'>
        Это приложение позволяет анализировать данные и получать рекомендации для спортсменов.
        Сервис будет прогнозировать прогресс и возможное выполнение тех или иных элементов программы по истории предыдущих выступлений и выполнения элементов на соревнованиях.
    </p>
    </p>
    """,unsafe_allow_html=True
)

st.sidebar.header("Параметры")
unit = st.sidebar.selectbox("Выберите id спортсмена", sorted(df_grouped2['unit_id'].unique()))
n_last = st.sidebar.slider("Количество последних записей для предсказания", 1, 10, 3)
n_elements = st.sidebar.slider("Количество рекомендованных элементов/похожих спортсменов", 1, 10, 3)

# Кнопка "Запустить анализ" в сайдбаре
if st.sidebar.button("Запустить анализ"):

    # Устанавливаем состояние, что кнопка была нажата
    st.session_state.analyze_clicked = True

# Если кнопка нажата, скрываем DataFrame и выполняем анализ
if st.session_state.get("analyze_clicked"):
    st.markdown(f"""<h1 style='text-align: center;'>Рекомендательная система для спортсмена unit = {unit}</h1>""",unsafe_allow_html=True)
    # Используем колонки для разделения информации
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Предсказания модели")
        real, binary = model__(df_grouped2, unit, n_last)
        st.write('**Элементы, которые лучше выполнить спортсмену**')
        st.success(f"**{'   '.join(binary)}**")

        st.subheader("👥 Похожие спортсмены")
        similar_units, recommended_elements = recommendation(elements, unit, n_elements)

        # Отображение похожих спортсменов
        st.dataframe(pd.DataFrame(similar_units[:n_elements], columns=["ID спортсмена"]).T, use_container_width=True,
                     hide_index=True)

        # График среднего общего балла по сезонам
        data = df_grouped[df_grouped['unit_id'] == unit][['season', 'overall_total_score']]
        aggregated_data = data.groupby('season', as_index=False)['overall_total_score'].mean()

        # Создаем график с Plotly
        fig_bar = px.bar(
            aggregated_data,
            x='season',
            y='overall_total_score',
            color='season',
            title=f'Средний общий балл по сезонам для спортсмена id={unit}',
            labels={'season': 'Сезон', 'overall_total_score': 'Средний общий балл'},
            text='overall_total_score',  # Отображаем значения на столбцах
            template='plotly_white'  # Стиль графика
        )

        # Настройка внешнего вида
        fig_bar.update_traces(
            textposition='outside',
            marker_line_color='black',
            marker_line_width=1.5  # Настройка текста и границ столбцов
        )
        fig_bar.update_layout(
            xaxis_title="Сезон",
            yaxis_title="Средний общий балл",
            legend_title="Сезон",
            showlegend=False  # Скрываем легенду, так как сезоны уже на оси X
        )

        # Отображение столбчатой диаграммы в Streamlit
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.subheader("🌟 Рекомендованные элементы")
        st.dataframe(
            pd.DataFrame(recommended_elements, columns=["Элемент"]).T,
            use_container_width=True,hide_index=True
        )
        st.subheader("🏆 Самый дорогой выполненный элемент")
        best = best_el(unit)
        st.dataframe(
            best[['element', 'base']].rename(columns={"element": "Элемент", "base": "Балл"}),
            use_container_width=True,hide_index=True
        )

        # Фильтруем данные для конкретного юнита
        df = df_grouped[df_grouped['unit_id'] == unit]

        # Извлекаем столбцы, которые заканчиваются на '_element_perfect'
        elements_col = [col for col in df.columns if col.endswith('_element_perfect')]

        # Суммируем значения по каждому элементу
        perfect_elements_sum = df[elements_col].sum()

        # Фильтруем элементы, где сумма больше 0
        perfect_elements_filtered = perfect_elements_sum[perfect_elements_sum > 0]

        # Преобразуем результат в DataFrame для удобства
        perfect_elements_df = perfect_elements_filtered.reset_index()
        perfect_elements_df.columns = ['element', 'perfect_count']

        # Удаляем суффикс '_element_perfect' из названий элементов
        perfect_elements_df['element'] = perfect_elements_df['element'].str.replace('_element_perfect', '')

        # Создаем круговую диаграмму с Plotly Express
        fig_pie = px.pie(
            perfect_elements_df,
            values='perfect_count',  # Значения для диаграммы
            names='element',  # Названия элементов
            title="Идеально выполненные элементы",  # Заголовок
            color_discrete_sequence=px.colors.qualitative.Pastel  # Цветовая схема
        )

        # Настройка внешнего вида
        fig_pie.update_traces(
            textposition='inside',  # Текст внутри секторов
            textinfo='percent+label'  # Отображать проценты и названия
        )
        fig_pie.update_layout(
            showlegend=False,  # Скрыть легенду
            margin=dict(l=20, r=20, t=40, b=20)  # Отступы
        )

        # Отображение круговой диаграммы в Streamlit
        st.plotly_chart(fig_pie, use_container_width=True)

# Если кнопка не нажата, показываем DataFrame
else:
    st.markdown(

        f"""<p style='text-align: center; color: #666; font-style: italic;'>
                Чтобы начать, выберите <strong>юнита (unit)</strong> в боковой панели слева и нажмите кнопку <strong>"Запустить анализ"</strong>.
            </p>""",
        unsafe_allow_html=True
    )
    st.header('Описание')
    st.write('Данный сайт позволяет предсказывать лучшие элементы для спортсмена, показывает их статистику по сезонам, '
             'элементы которые он выполнял чаще всего идеально и многое другое')
    st.markdown(
        """
        <p style='color: #666; font-size: 14px; font-style: italic;'>
            Качество предсказаний зависит от количества юнитов в датасете - чем их больше, тем точнее предсказания. Это демо версия с ограниченным числом юнитов.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.header("📂 Данные")
    st.write(f'Всего юнитов в датасете: {len(df_grouped['unit_id'].unique())}')
    st.dataframe(df_grouped.head())
    with st.expander("📋 Показать список элементов"):
        st.markdown(f"""<p style='color: #666; font-style: italic;'>Тут вы можете посмотреть полное название элементов.</p>""",unsafe_allow_html=True)
        st.write("Список элементов фигурного катания:")
        st.dataframe(styled_df, hide_index=True)  # Отображаем стилизованный DataFrame
