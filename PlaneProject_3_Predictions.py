import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

label_encoder = LabelEncoder()

years = [2018, 2019, 2020, 2021, 2022]

for year in years:
	print("----------", year, "------------")
	df = pd.read_parquet(f"./data/clean_joined_{year}.parquet").dropna()
	print(df.columns)
	# print(df.head())

	# plt.figure(figsize=(10,8))
	# sns.heatmap(df.corr(method='pearson', numeric_only=True), annot=True)

	df_current = df.copy()
	df_current["MFR"] = label_encoder.fit_transform(df_current["MFR"])

	X = df_current.drop(['MFR', 'Tail_Number', 'Month', 'N-NUMBER', 'TYPE-ACFT'], axis=1)
	# X = df_current.drop(['MFR', 'Tail_Number', 'Month', 'N-NUMBER', 'TYPE-ACFT', 'YEAR MFR', 'SPEED', 'NO-SEATS', 'NO-ENG'], axis=1)

	y = df_current.MFR

	scores = {
		'gnb' : [],
		'rfc' : [],
		'dt'  : [],
		'ada' : []
	}

	gnb = GaussianNB()
	rfc = RandomForestClassifier(max_depth=2, random_state=42, n_jobs=-1)
	dt = DecisionTreeClassifier(max_depth=5, random_state=42)
	ada = AdaBoostClassifier(random_state=42)

	sample_sizes = []
	sample_sizes.extend(np.arange(1_000, 10_000, 1_000, dtype=int))
	sample_sizes.extend(np.arange(10_000, 100_000, 10_000, dtype=int))
	sample_sizes.extend(np.arange(100_000, 1_000_000, 100_000, dtype=int))
	sample_sizes.extend(np.arange(1_000_000, len(df), 1_000_000, dtype=int))

	for size in sample_sizes:
		print(size)
		_X = X[:size]
		_y = y[:size]

		X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size = 0.33, random_state = 42)

		# Gaussian Naive Bayes
		gnb.fit(X_train, y_train)
		scores['gnb'].append(gnb.score(X_test, y_test))

		# Random Forest
		rfc.fit(X_train, y_train)
		scores['rfc'].append(rfc.score(X_test, y_test))

		# Decision Tree
		dt.fit(X_train, y_train)
		scores['dt'].append(dt.score(X_test, y_test))

		# Adaboost
		ada.fit(X_train, y_train)
		scores['ada'].append(ada.score(X_test, y_test))


	for _type, _score in scores.items():
		print(_type, _score)


	plt.figure(figsize=(4,6))

	plt.plot(sample_sizes, scores['gnb'], label="GNB")
	plt.plot(sample_sizes, scores['rfc'], label="RFC")
	plt.plot(sample_sizes, scores['dt'], label="DT")
	plt.plot(sample_sizes, scores['ada'], label="AdaBoost")

	plt.legend()
	plt.xlabel("Sample Size")
	plt.ylabel("Score")
	plt.title(f"Classifier Performance ({year})")
	plt.show()

	sc_df = pd.DataFrame.from_dict(scores, orient='index')
	print(sc_df)
	sc_df.to_parquet(f"./data/predictions_{year}.parquet")
	plt.savefig(f"./data/predictions_{year}.png")
