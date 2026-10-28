# Online Shoppers Purchasing Intention

Источник: Sakar, C. & Kastro, Y. (2018), UCI, https://doi.org/10.24432/C5F88Q.
Лицензия: CC BY 4.0. CSV взят из официального архива UCI (см. provenance.json).
SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

Размер: 12330 строк, 17 признаков + Revenue.
Целевая: 1908 покупок (15.47%).
Пропусков: 0; полных дубликатов: 125.
Дубликаты удалены до разбиения. Train: 9764; test: 2441.
Seed: 42, stratify=y. EDA проводится на train.

| Поле | Смысл | Тип | Пропуски |
|---|---|---|---|
| Administrative | Административные страницы | int64 | 0.0% |
| Administrative_Duration | Время административных страниц, с | float64 | 0.0% |
| Informational | Информационные страницы | int64 | 0.0% |
| Informational_Duration | Время информационных страниц, с | float64 | 0.0% |
| ProductRelated | Товарные страницы | int64 | 0.0% |
| ProductRelated_Duration | Время товарных страниц, с | float64 | 0.0% |
| BounceRates | Показатель отказов | float64 | 0.0% |
| ExitRates | Показатель выходов | float64 | 0.0% |
| PageValues | Аналитическая ценность страниц | float64 | 0.0% |
| SpecialDay | Близость особого дня | float64 | 0.0% |
| Month | Месяц | object | 0.0% |
| OperatingSystems | Код ОС | int64 | 0.0% |
| Browser | Код браузера | int64 | 0.0% |
| Region | Код региона | int64 | 0.0% |
| TrafficType | Код трафика | int64 | 0.0% |
| VisitorType | Тип посетителя | object | 0.0% |
| Weekend | Выходной | bool | 0.0% |
| Revenue | Покупка (целевая) | bool | 0.0% |

Технические коды считаются номинальными категориями. Полной даты, текстов и координат нет.
PageValues необходимо проверять на доступность до покупки; производный AvgPageValue наследует риск.
