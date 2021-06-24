
import math
import sys
import pandas as pd
from decison_tree import decision_btree
from file_healper import file_operation
def classify_validation_data(cls_validation_data):
	classification_result = list()
	for idx in cls_validation_data.index:
		if cls_validation_data["TailLn"][idx]<=9.0:
			if cls_validation_data["HairLn"][idx]<=8.0:
				if cls_validation_data["TailLn"][idx]<=6.0:
					print(-1)
					classification_result.append(-1)
				else: 
					if cls_validation_data["BangLn"][idx]<=4.0:
						print(1)
						classification_result.append(1)
					else: 
						print(-1)
						classification_result.append(-1)
			else: 
				print(-1)
				classification_result.append(-1)
		else: 
			if cls_validation_data["HairLn"][idx]<=9.0:
				if cls_validation_data["BangLn"][idx]<=5.0:
					print(1)
					classification_result.append(1)
				else: 
					if cls_validation_data["TailLn"][idx]<=15.0:
						if cls_validation_data["Ht"][idx]<=134.0:
							print(1)
							classification_result.append(1)
						else: 
							print(-1)
							classification_result.append(-1)
					else: 
						print(1)
						classification_result.append(1)
			else: 
				if cls_validation_data["BangLn"][idx]<=4.0:
					if cls_validation_data["HairLn"][idx]<=11.0:
						print(1)
						classification_result.append(1)
					else: 
						if cls_validation_data["Ht"][idx]<=134.0:
							print(1)
							classification_result.append(1)
						else: 
							print(-1)
							classification_result.append(-1)
				else: 
					if cls_validation_data["TailLn"][idx]<=16.0:
						print(-1)
						classification_result.append(-1)
					else: 
						if cls_validation_data["HairLn"][idx]<=10.0:
							print(1)
							classification_result.append(1)
						else: 
							print(-1)
							classification_result.append(-1)
	return classification_result
def main():
	file_obj = file_operation()
	cls_validation_data = file_obj.read_csv("Abominable_VALIDATION_Data.csv")
	classification_result = classify_validation_data(cls_validation_data)
	cls_validation_data.insert(8, "cls_result", classification_result, True)
	cls_validation_data.to_csv('MyClassifications.csv')

if __name__ == '__main__':
	main()
