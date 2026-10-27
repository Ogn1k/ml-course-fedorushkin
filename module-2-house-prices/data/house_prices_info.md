# House Prices — описание датасета

Источник: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
Зеркало загрузки: https://github.com/Rswcf/houseprices_kaggle
URL файлов и SHA-256 записаны в provenance.json.

Основа — Ames Housing, Dean De Cock (2011), продажи Эймса, Айова, 2006–2010.
Исходный Ames: 2930 наблюдений, версия Kaggle: 1460 train + 1459 test.
Train содержит 81 столбец (79 признаков, Id, SalePrice); test — 80 без SalePrice.
По dtype признаков: 36 числовых, 43 строковых. Порядковые шкалы определяются по смыслу.
Площадь измерена в квадратных футах, LotFrontage — в футах, цены — в долларах.

Полный словарь каждого столбца: data_description.txt.
None — реальная категория MasVnrType; NA в ряде характеристик означает отсутствие объекта.
CSV читается с keep_default_na=False и na_values=['NA', ''], чтобы не потерять None.
Статистические заполнения учатся только внутри обучающих фолдов.

## Статистика цели (весь train, только справочная)

Среднее: $180,921.20; медиана: $163,000; диапазон: $34,900–$755,000; skew: 1.8829; эксцесс: 6.5363.

## Поля, типы и фактические пропуски

| Поле | dtype | NA | NA, % |
|---|---|---:|---:|
| Id | int64 | 0 | 0.00 |
| MSSubClass | int64 | 0 | 0.00 |
| MSZoning | object | 0 | 0.00 |
| LotFrontage | float64 | 259 | 17.74 |
| LotArea | int64 | 0 | 0.00 |
| Street | object | 0 | 0.00 |
| Alley | object | 1369 | 93.77 |
| LotShape | object | 0 | 0.00 |
| LandContour | object | 0 | 0.00 |
| Utilities | object | 0 | 0.00 |
| LotConfig | object | 0 | 0.00 |
| LandSlope | object | 0 | 0.00 |
| Neighborhood | object | 0 | 0.00 |
| Condition1 | object | 0 | 0.00 |
| Condition2 | object | 0 | 0.00 |
| BldgType | object | 0 | 0.00 |
| HouseStyle | object | 0 | 0.00 |
| OverallQual | int64 | 0 | 0.00 |
| OverallCond | int64 | 0 | 0.00 |
| YearBuilt | int64 | 0 | 0.00 |
| YearRemodAdd | int64 | 0 | 0.00 |
| RoofStyle | object | 0 | 0.00 |
| RoofMatl | object | 0 | 0.00 |
| Exterior1st | object | 0 | 0.00 |
| Exterior2nd | object | 0 | 0.00 |
| MasVnrType | object | 8 | 0.55 |
| MasVnrArea | float64 | 8 | 0.55 |
| ExterQual | object | 0 | 0.00 |
| ExterCond | object | 0 | 0.00 |
| Foundation | object | 0 | 0.00 |
| BsmtQual | object | 37 | 2.53 |
| BsmtCond | object | 37 | 2.53 |
| BsmtExposure | object | 38 | 2.60 |
| BsmtFinType1 | object | 37 | 2.53 |
| BsmtFinSF1 | int64 | 0 | 0.00 |
| BsmtFinType2 | object | 38 | 2.60 |
| BsmtFinSF2 | int64 | 0 | 0.00 |
| BsmtUnfSF | int64 | 0 | 0.00 |
| TotalBsmtSF | int64 | 0 | 0.00 |
| Heating | object | 0 | 0.00 |
| HeatingQC | object | 0 | 0.00 |
| CentralAir | object | 0 | 0.00 |
| Electrical | object | 1 | 0.07 |
| 1stFlrSF | int64 | 0 | 0.00 |
| 2ndFlrSF | int64 | 0 | 0.00 |
| LowQualFinSF | int64 | 0 | 0.00 |
| GrLivArea | int64 | 0 | 0.00 |
| BsmtFullBath | int64 | 0 | 0.00 |
| BsmtHalfBath | int64 | 0 | 0.00 |
| FullBath | int64 | 0 | 0.00 |
| HalfBath | int64 | 0 | 0.00 |
| BedroomAbvGr | int64 | 0 | 0.00 |
| KitchenAbvGr | int64 | 0 | 0.00 |
| KitchenQual | object | 0 | 0.00 |
| TotRmsAbvGrd | int64 | 0 | 0.00 |
| Functional | object | 0 | 0.00 |
| Fireplaces | int64 | 0 | 0.00 |
| FireplaceQu | object | 690 | 47.26 |
| GarageType | object | 81 | 5.55 |
| GarageYrBlt | float64 | 81 | 5.55 |
| GarageFinish | object | 81 | 5.55 |
| GarageCars | int64 | 0 | 0.00 |
| GarageArea | int64 | 0 | 0.00 |
| GarageQual | object | 81 | 5.55 |
| GarageCond | object | 81 | 5.55 |
| PavedDrive | object | 0 | 0.00 |
| WoodDeckSF | int64 | 0 | 0.00 |
| OpenPorchSF | int64 | 0 | 0.00 |
| EnclosedPorch | int64 | 0 | 0.00 |
| 3SsnPorch | int64 | 0 | 0.00 |
| ScreenPorch | int64 | 0 | 0.00 |
| PoolArea | int64 | 0 | 0.00 |
| PoolQC | object | 1453 | 99.52 |
| Fence | object | 1179 | 80.75 |
| MiscFeature | object | 1406 | 96.30 |
| MiscVal | int64 | 0 | 0.00 |
| MoSold | int64 | 0 | 0.00 |
| YrSold | int64 | 0 | 0.00 |
| SaleType | object | 0 | 0.00 |
| SaleCondition | object | 0 | 0.00 |
| SalePrice | int64 | 0 | 0.00 |
