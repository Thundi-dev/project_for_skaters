# Описание проекта
Сервис  “Мой Чемпион” помогает спортивным школам фигурного катания, тренерам мониторить результаты своих подопечных и планировать дальнейшее развитие спортсменов.
## Цель
Создать модель, помогающую находить элементы, которые могут быть успешно исполнены спортсменом на соревновании. 
Сервис будет прогнозировать прогресс и возможное выполнение тех или иных элементов программы по истории предыдущих выступлений и выполнения элементов на соревнованиях.

# Описание признаков

**df_grouped2** - сгруппированная таблица с паспарсенными элементами

- `unit_id`: идентификатор юнита
- `color`: категория
- `school_id`: идентификатор школы
- `tournament_id`: идентификатор турнира
- `date_start`: дата начала
- `date_end`: дата завершения
- `origin_id`: место проведения
- `elements_score`: реальная оценка всех выполненных элементов, base_score+goe
- `place`: занятое место в категории category_name+segment_name
- `segment_name`: название сегмента
- `overall_place`: итоговое место в турнире
- `overall_total_score`: итоговая оценка за весь турнир
- `season`: сезон соревнований
- `starting_place`: начальле место

**Описание элементов**
Прыжки:
- `2A_element_perfect`: Двойной аксель.
- `3A_element_perfect`: Тройной аксель.
  
- `2F_element_perfect`: Двойной флип.
- `3F_element_perfect`: Тройной флип.
- `4F_element_perfect`: Четверной флип.
  
- `2Lo_element_perfect`: Двойной луп.
- `3Lo_element_perfect`: Тройной луп.
- `4Lo_element_perfect`: Четверной луп.
  
- `2Lz_element_perfect`: Двойной лутц.
- `3Lz_element_perfect`: Тройной лутц.
- `4Lz_element_perfect`: Четверной лутц.
  
- `2S_element_perfect`: Двойной сальхов.
- `3S_element_perfect`: Тройной сальхов.
- `4S_element_perfect`: Четверной сальхов.

`2T_element_perfect`: Двойной тулуп.
`3T_element_perfect`: Тройной тулуп.
`4T_element_perfect`: Четверной тулуп.

- `CCSp2_element_perfect`: Каскадное вращение 2 уровня.
- `CCSp3_element_perfect`: Каскадное вращение 3 уровня.
- `CCSp4_element_perfect`: Каскадное вращение 4 уровня.

- `CCoSp2_element_perfect`: Комбинированное вращение 2 уровня.
- `CCoSp3_element_perfect`: Комбинированное вращение 3 уровня.
- `CCoSp4_element_perfect`: Комбинированное вращение 4 уровня.
- `CCoSpB_element_perfect`: Комбинированное вращение базового уровня.

- `CSSp2_element_perfect`: Сит-спин 2 уровня.
- `CSSp3_element_perfect`: Сит-спин 3 уровня.
- `CSSp4_element_perfect`: Сит-спин 4 уровня.
- `CSSpB_element_perfect`: Сит-спин базового уровня.

- `CSp2_element_perfect`: Вращение в ласточке 2 уровня.
- `CSp3_element_perfect`: Вращение в ласточке 3 уровня.
- `CSp4_element_perfect`: Вращение в ласточке 4 уровня.
- `CSpB_element_perfect`: Вращение в ласточке базового уровня.

`FCCSp4_element_perfect`: летящего каскадного вращения 4 уровня (Flying Camel Spin).

- `FCCoSp2_element_perfect`: Летящее комбинированное вращение 2 уровня.
- `FCCoSp3_element_perfect`: Летящее комбинированное вращение 3 уровня.
- `FCCoSp4_element_perfect`: Летящее комбинированное вращение 4 уровня.
- `FCCoSpB_element_perfect`: Летящее комбинированное вращение базового уровня.

- `FCSSp2_element_perfect`: Летящий сит-спин 2 уровня.
- `FCSSp3_element_perfect`: Летящий сит-спин 3 уровня.
- `FCSSp4_element_perfect`: Летящий сит-спин 4 уровня.
- `FCSSpB_element_perfect`: Летящий сит-спин базового уровня.

- `FCSp2_element_perfect`: Летящее вращение в ласточке 2 уровня.
- `FCSp3_element_perfect`: Летящее вращение в ласточке 3 уровня.
- `FCSp4_element_perfect`: Летящее вращение в ласточке 4 уровня.
- `FCSpB_element_perfect`: Летящее вращение в ласточке базового уровня.

- `FLSp2_element_perfect`: Летящее вращение ласточкой 2 уровня.
- `FLSp3_element_perfect`: Летящее вращение ласточкой 3 уровня.
- `FLSp4_element_perfect`: Летящее вращение ласточкой 4 уровня.

- `FSSp2_element_perfect`: Летящий сит-спин 2 уровня.
- `FSSp3_element_perfect`: Летящий сит-спин 3 уровня.
- `FSSp4_element_perfect`: Летящий сит-спин 4 уровня.
- `FSSpB_element_perfect`: Летящий сит-спин базового уровня.

`FSsp2_element_perfect`: летящих вращений сит-спин (Flying Sit Spin) 2 уровня.

- `LSp2_element_perfect`: Вращение ласточкой 2 уровня.
- `LSp3_element_perfect`: Вращение ласточкой 3 уровня.
- `LSp4_element_perfect`: Вращение ласточкой 4 уровня.

- `SSp2_element_perfect`: Сит-спин 2 уровня.
- `SSp3_element_perfect`: Сит-спин 3 уровня.
- `SSp4_element_perfect`: Сит-спин 4 уровня.
- `SSpB_element_perfect`: Сит-спин базового уровня.

`USpB_element_perfect`: вращений вверх (Upright Spin) базового уровня.

Дорожки шагов:

- `StSq2_element_perfect`: Дорожка шагов 2 уровня.
- `StSq3_element_perfect`: Дорожка шагов 3 уровня.
- `StSq4_element_perfect`: Дорожка шагов 4 уровня.
- `StSqB_element_perfect`: Дорожка шагов базового уровня.
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
