
# Описание проекта
Добро пожаловать в сервис для анализа и прогнозирования успешности элементов фигурного катания!
Наш инструмент помогает тренерам и спортивным школам отслеживать прогресс спортсменов, анализировать их выступления и получать персонализированные рекомендации для улучшения результатов.

С помощью моделей машинного обучения мы предсказываем, какие элементы спортсмен может успешно выполнить на соревнованиях, а также предлагаем новые элементы для освоения на основе опыта похожих спортсменов.

Начните анализ, выбрав спортсмена и параметры для прогнозирования, и получите детальную статистику и рекомендации для достижения новых высот!😊

**Ссылка на сайт: [Рекомендательная система для спортсменов](https://projectforskaters-fpuycatpiaykyatfdivzez.streamlit.app)**

## Описание файлов
1) Папка [notebook](notebooks) содержит саму тетрадку с выполненным проектом- [go_project_by_Sazanova](notebooks/go_project_by_Sazanova.ipynb) и описание проекта 
2) [app_skate](app_skate.py) - само приложение
3) [df_grouped](df_grouped.csv),[scores](scores.csv)  - содержит используемый датасеты
4) `requirements.txt` - содержит импортирумые библиотеки 
5) `wallpaper_0201340-optimized.jpg` - фоновая картинка

# Описание признаков

**df_grouped** - сгруппированная таблица с паспарсенными элементами

| Элемент                     | Описание                                                                 |
|-----------------------------|--------------------------------------------------------------------------|
| `unit_id`                   | Идентификатор юнита                                                     |
| `color`                     | Категория                                                               |
| `school_id`                 | Идентификатор школы                                                     |
| `tournament_id`             | Идентификатор турнира                                                   |
| `date_start`                | Дата начала                                                             |
| `date_end`                  | Дата завершения                                                         |
| `origin_id`                 | Место проведения                                                        |
| `elements_score`            | Реальная оценка всех выполненных элементов (base_score + goe)           |
| `place`                     | Занятое место в категории (category_name + segment_name)                |
| `segment_name`              | Название сегмента                                                       |
| `overall_place`             | Итоговое место в турнире                                                |
| `overall_total_score`       | Итоговая оценка за весь турнир                                          |
| `season`                    | Сезон соревнований                                                      |
| `starting_place`            | Начальное место                                                         |
| `A_element_perfect`         | Аксель                                                                  |
| `CCSpB_element_perfect`     | Каскадное вращение базового уровня                                      |
| `CCSp_element_perfect`      | Каскадное вращение                                                      |
| `CCoSp_element_perfect`     | Комбинированное вращение                                                |
| `CSSp_element_perfect`      | Сит-спин                                                                |
| `CSp_element_perfect`       | Вращение в ласточке                                                     |
| `ChS_element_perfect`       | Вращение в крюке                                                        |
| `ChSpl_element_perfect`     | Вращение в крюке с переходом                                            |
| `CoSp2_element_perfect`     | Комбинированное вращение 2 уровня                                       |
| `CoSpB_element_perfect`     | Комбинированное вращение базового уровня                                |
| `CoSp_element_perfect`      | Комбинированное вращение                                                |
| `Eu_element_perfect`        | Вращение в ласточке с переходом                                         |
| `FCCoSp_element_perfect`    | Летящее комбинированное вращение                                        |
| `FCSSp_element_perfect`     | Летящий сит-спин                                                        |
| `FCSp_element_perfect`      | Летящее вращение в ласточке                                             |
| `FCoSp2_element_perfect`    | Летящее комбинированное вращение 2 уровня                               |
| `FCoSp_element_perfect`     | Летящее комбинированное вращение                                        |
| `FSSp_element_perfect`      | Летящий сит-спин                                                        |
| `FUSp_element_perfect`      | Летящее вращение вверх                                                  |
| `F_element_perfect`         | Флип                                                                    |
| `LSpB_element_perfect`      | Вращение в ласточке базового уровня                                     |
| `LSp_element_perfect`       | Вращение в ласточке                                                     |
| `Lo_element_perfect`        | Луп                                                                     |
| `Lz_element_perfect`        | Лутц                                                                    |
| `SSp_element_perfect`       | Сит-спин                                                                |
| `S_element_perfect`         | Сальхов                                                                 |
| `Sp_element_perfect`        | Вращение                                                                |
| `StSq_element_perfect`      | Дорожка шагов                                                           |
| `T_element_perfect`         | Тулуп                                                                   |
| `W_element_perfect`         | Вращение в ласточке с переходом                                         |
| `2A_element_perfect`        | Двойной аксель                                                          |
| `3A_element_perfect`        | Тройной аксель                                                          |
| `2F_element_perfect`        | Двойной флип                                                            |
| `3F_element_perfect`        | Тройной флип                                                            |
| `4F_element_perfect`        | Четверной флип                                                          |
| `2Lo_element_perfect`       | Двойной луп                                                             |
| `3Lo_element_perfect`       | Тройной луп                                                             |
| `4Lo_element_perfect`       | Четверной луп                                                           |
| `2Lz_element_perfect`       | Двойной лутц                                                            |
| `3Lz_element_perfect`       | Тройной лутц                                                            |
| `4Lz_element_perfect`       | Четверной лутц                                                          |
| `2S_element_perfect`        | Двойной сальхов                                                         |
| `3S_element_perfect`        | Тройной сальхов                                                         |
| `4S_element_perfect`        | Четверной сальхов                                                       |
| `2T_element_perfect`        | Двойной тулуп                                                           |
| `3T_element_perfect`        | Тройной тулуп                                                           |
| `4T_element_perfect`        | Четверной тулуп                                                         |
| `CCSp2_element_perfect`     | Каскадное вращение 2 уровня                                             |
| `CCSp3_element_perfect`     | Каскадное вращение 3 уровня                                             |
| `CCSp4_element_perfect`     | Каскадное вращение 4 уровня                                             |
| `CCoSp2_element_perfect`    | Комбинированное вращение 2 уровня                                       |
| `CCoSp3_element_perfect`    | Комбинированное вращение 3 уровня                                       |
| `CCoSp4_element_perfect`    | Комбинированное вращение 4 уровня                                       |
| `CCoSpB_element_perfect`    | Комбинированное вращение базового уровня                                |
| `CSSp2_element_perfect`     | Сит-спин 2 уровня                                                       |
| `CSSp3_element_perfect`     | Сит-спин 3 уровня                                                       |
| `CSSp4_element_perfect`     | Сит-спин 4 уровня                                                       |
| `CSSpB_element_perfect`     | Сит-спин базового уровня                                                |
| `CSp2_element_perfect`      | Вращение в ласточке 2 уровня                                            |
| `CSp3_element_perfect`      | Вращение в ласточке 3 уровня                                            |
| `CSp4_element_perfect`      | Вращение в ласточке 4 уровня                                            |
| `CSpB_element_perfect`      | Вращение в ласточке базового уровня                                     |
| `FCCSp4_element_perfect`    | Летящее каскадное вращение 4 уровня                                     |
| `FCCoSp2_element_perfect`   | Летящее комбинированное вращение 2 уровня                               |
| `FCCoSp3_element_perfect`   | Летящее комбинированное вращение 3 уровня                               |
| `FCCoSp4_element_perfect`   | Летящее комбинированное вращение 4 уровня                               |
| `FCCoSpB_element_perfect`   | Летящее комбинированное вращение базового уровня                        |
| `FCSSp2_element_perfect`    | Летящий сит-спин 2 уровня                                               |
| `FCSSp3_element_perfect`    | Летящий сит-спин 3 уровня                                               |
| `FCSSp4_element_perfect`    | Летящий сит-спин 4 уровня                                               |
| `FCSSpB_element_perfect`    | Летящий сит-спин базового уровня                                        |
| `FCSp2_element_perfect`     | Летящее вращение в ласточке 2 уровня                                    |
| `FCSp3_element_perfect`     | Летящее вращение в ласточке 3 уровня                                    |
| `FCSp4_element_perfect`     | Летящее вращение в ласточке 4 уровня                                    |
| `FCSpB_element_perfect`     | Летящее вращение в ласточке базового уровня                             |
| `FLSp2_element_perfect`     | Летящее вращение ласточкой 2 уровня                                     |
| `FLSp3_element_perfect`     | Летящее вращение ласточкой 3 уровня                                     |
| `FLSp4_element_perfect`     | Летящее вращение ласточкой 4 уровня                                     |
| `FSSp2_element_perfect`     | Летящий сит-спин 2 уровня                                               |
| `FSSp3_element_perfect`     | Летящий сит-спин 3 уровня                                               |
| `FSSp4_element_perfect`     | Летящий сит-спин 4 уровня                                               |
| `FSSpB_element_perfect`     | Летящий сит-спин базового уровня                                        |
| `FSsp2_element_perfect`     | Летящие вращения сит-спин 2 уровня                                      |
| `LSp2_element_perfect`      | Вращение ласточкой 2 уровня                                             |
| `LSp3_element_perfect`      | Вращение ласточкой 3 уровня                                             |
| `LSp4_element_perfect`      | Вращение ласточкой 4 уровня                                             |
| `SSp2_element_perfect`      | Сит-спин 2 уровня                                                       |
| `SSp3_element_perfect`      | Сит-спин 3 уровня                                                       |
| `SSp4_element_perfect`      | Сит-спин 4 уровня                                                       |
| `SSpB_element_perfect`      | Сит-спин базового уровня                                                |
| `USpB_element_perfect`      | Вращение вверх базового уровня                                          |
| `StSq2_element_perfect`     | Дорожка шагов 2 уровня                                                  |
| `StSq3_element_perfect`     | Дорожка шагов 3 уровня                                                  |
| `StSq4_element_perfect`     | Дорожка шагов 4 уровня                                                  |
| `StSqB_element_perfect`     | Дорожка шагов базового уровня                                           |

----
# **Модели**
## **Модель мультиклассификации**

Сначала отсортирвоали значения по количеству соревнований у каждого человека и удалим все что меньше 4.

*Описание модели мультиклассификации для прогнозирования лучших элементов*

*Цель модели:*

Модель предназначена для прогнозирования лучших элементов для спортсменов на основе их предыдущих выступлений. Модель анализирует исторические данные и предсказывает, какие элементы будут наиболее успешными в последних N выступлениях конкретного спортсмена.

*Как работает модель:*
    
1) *Выбор спортсмена:*
    - Вам предоставляется список всех unit_id спортсменов. Вы можете выбрать любого спортсмена, для которого хотите получить прогноз
2) *Прогнозирование для последних N выступлений:*
    - Модель анализирует данные последних N выступлений выбранного спортсмена.
    - На основе этих данных она предсказывает, какие элементы будут наиболее успешными.
3) *Результаты:*
    - Модель возвращает два набора данных(Каждый выводится единым списком за N дней):
        - Реальные значения: Фактические элементы, которые спортсмен выполнил в последних N выступлениях. 
        - Прогнозируемые значения: Элементы, которые модель рекомендует как наиболее успешные.

Средняя точность по всем элементам: 0.7142857142857143

## **Рекомендательная модель**

*Описание рекомендационной модели для прогнозирования лучших элементов*

*Цель модели:*

Рекомендационная модель предназначена для предоставления персонализированных рекомендаций спортсменам на основе их текущих навыков и схожести с другими спортсменами. Модель анализирует данные о выполненных элементах и рекомендует новые элементы, которые могут быть успешно освоены, основываясь на опыте похожих спортсменов.

*Как работает модель:*

1) *Выбор спортсмена:*

Вам предоставляется список всех unit_id спортсменов. Вы можете выбрать любого спортсмена, для которого хотите получить рекомендации.

2) *Анализ текущих навыков и предсказания*

Модель анализирует элементы, которые выбранный спортсмен уже успешно выполняет (например, элементы с положительными значениями в данных). Модель вычисляет схожесть между выбранным спортсменом и всеми остальными спортсменами на основе их выполнения элементов. Для этого используется косинусная схожесть. Наиболее похожие спортсмены определяются как те, у которых схожесть с выбранным спортсменом максимальна.

4) *Рекомендация новых элементов:*

Модель возвращает список элементов, которые могут быть успешно освоены выбранным спортсменом, основываясь на опыте похожих спортсменов.

5) *Результаты:*

Модель возвращает два набора данных:

- Похожие спортсмены:

    - Список unit_id спортсменов, которые наиболее схожи с выбранным спортсменом по выполнению элементов.

- Рекомендованные элементы:

    - Список элементов, которые похожие спортсмены успешно выполняют, но которые ещё не освоены выбранным спортсменом. Эти элементы рекомендуются для освоения.
 
----
# Инструкция по запуску (применению)

## Рекомендательная система для спортсменов

**Ссылка на сайт: [Рекомендательная система для спортсменов](https://projectforskaters-fpuycatpiaykyatfdivzez.streamlit.app)**

**Это приложение позволяет анализировать данные и получать рекомендации для спортсменов. Сервис прогнозирует прогресс и возможное выполнение элементов программы на основе истории выступлений.**

---

## Описание

Данный сайт позволяет:
- Предсказывать лучшие элементы для спортсмена.
- Показывать элементы, которые спортсмен выполнял чаще всего идеально.
- Анализировать данные и предоставлять рекомендации для улучшения результатов.
- Так же увидеть анализ по сезонам в виде графика

Нюансы:
- Качество предсказаний зависит от количества юнитов в датасете - чем их больше, тем точнее предсказания. Это демо версия с ограниченным числом юнитов.
- В будущем добавится версия с добавлением строчек вм датасет, которые могут повлиять на предсказания

Как работает:
1) Пользователь заходит на сайт и в боковой панели выбирает: id спортсмена, количество последних записей для предсказания, количество рекомендованных элементов/похожих спортсменов.
2) Пользователь нажимает на кнопку анализировать
3) После нажатия появляется статистика.

---
### Начальный экран

![image](https://github.com/user-attachments/assets/6a1b1aba-7952-4b4b-b2a8-d2a2b17a238e) 
*Рис. 1*

![image](https://github.com/user-attachments/assets/f299ba6e-1f5c-409e-a201-db572eebcb53)
*Рис. 2*

![image](https://github.com/user-attachments/assets/c564bc96-75f5-46ed-9e5d-79fdef309d60)
*Рис. 3*

- **Пример данных**: Отображаются первые 4 строки датасета `df_grouped`.
- **Описание функционала**: Подробное описание возможностей приложения.
- **Всего юнитов в датасете**: Выводит количество юнитов
- **Список элементов**: При клике на элемент отображается его полное описание (тип прыжка, описание).

### Боковая панель "Параметры" 

*см. рис.1, рис.2, рис.3*

- **Выберите id спортсмена**: Выберите идентификатор юнита (`unit_id`) из списка.
- **Количество последних записей для предсказания**: Укажите, сколько последних записей использовать для анализа.
- **Количество рекомендованных элементов/похожих спортсменов**: Выберите, сколько элементов будет рекомендовано и сколько выведет похожих спортсменов(прямо зависит от того, сколько юнитов в датасете)
- **Запустить анализ**: Нажмите кнопку, чтобы начать анализ.
  
---
### Экран после активации кнопки

![image](https://github.com/user-attachments/assets/34fca71a-7166-42fa-b1c9-e0b90c5fffe2)
*Рис. 4*

После нажатия кнопки *Запустить анализ*. Вам открывается отображение вся статистика рекомендаций и достижений спортсмена

- **Предсказания модели**: Отображаются предсказания модели мультиклассификации.  Данный раздел показывает какие элементы спортсмену желательно выполнить.*Возможны ошибки при увеличении тестовой выборки,которую контролирует пользователь из-за несбалансированных классов.*
- **Похожие спортсмены**: Данный раздел показывает результаты рекомендательной модели. Показывает спортсменов похожих на юнита. Размер выборки зависит от пользователя. Они отсортированы по убиванию "похожести". 
- **Рекомендованные элементы**: Данный раздел показывает результаты рекомендательной модели. Он показывает какие элементы спортсмен может попробовать выполнить.Размер выборки зависит от пользователя. Они отсортированы по убиванию точности.
- **Самый дорогой выполненный элемент**: Выводит самый дорогой элемент,который выполнил пользователь
- **Идеально выполненные элементы**: Показывает в виде графика(pie) количество иделально выполненных элементов
- **Средний общий балл по сезонам для спортсмена**: Выводит среднее значение по сезонам 


