# Telco Customer Churn

Источник задания: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Скачано из официального репозитория IBM; полный URL и SHA-256 — в provenance.json.
Дата скачивания: 27.10.2026. Исходные байты не изменялись.

7043 строки, 21 столбец; 19 признаков, идентификатор и цель.
No: 5174 (73.46%); Yes: 1869 (26.54%).
Уникальные клиенты, 11 текстовых пропусков в TotalCharges; to_numeric(errors='coerce').
В первичном isna() текстовые пробелы не считаются NaN.
Три количественных признака, 16 категориальных (включая бинарный SeniorCitizen).

| Поле | Тип CSV | Смысл |
|---|---|---|
| customerID | object | Уникальный идентификатор; исключается из модели |
| gender | object | Пол |
| SeniorCitizen | int64 | Пожилой клиент, 0/1 |
| Partner | object | Наличие партнёра |
| Dependents | object | Наличие иждивенцев |
| tenure | int64 | Стаж обслуживания, месяцы |
| PhoneService | object | Телефония |
| MultipleLines | object | Несколько телефонных линий |
| InternetService | object | Тип интернет-доступа: DSL/Fiber optic/No |
| OnlineSecurity | object | Онлайн-безопасность |
| OnlineBackup | object | Резервное копирование |
| DeviceProtection | object | Защита устройств |
| TechSupport | object | Техническая поддержка |
| StreamingTV | object | Телевидение |
| StreamingMovies | object | Фильмы |
| Contract | object | Тип контракта |
| PaperlessBilling | object | Электронные счета |
| PaymentMethod | object | Способ оплаты |
| MonthlyCharges | float64 | Ежемесячный платёж |
| TotalCharges | object | Сумма платежей; 11 пустых строк |
| Churn | object | Целевая переменная: Yes=1, No=0 |

## Преобразования

`TotalCharges`: медиана train (в CV только обучающий fold).
`TenureYears = tenure / 12`; `AvgMonthlyCharge = TotalCharges / (tenure + 1)`;
`IsNewClient = (tenure <= 6)`; `HasMultipleServices` — число 9 услуг:
PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
Интернет считается подключённым при InternetService != 'No', прочие — при 'Yes'.
Категории кодируются LabelEncoder по train, неизвестные получают -1.
23 конечных признака. Единицы платежей оставлены как в учебном источнике.
Отсутствует временная история; пригодность для будущего реального оттока не подтверждена.
