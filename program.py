import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random

# Define the data
'''
Skin_Tone = {
0: "Fair",
1: "Medium",
2: "Dark"
}
Occasion = {
0: "Formal",
1: "Casual",
2: "Semi-formal"
}
Gender = {
0: "Male",
1: "Female",
}
Season = {
0: "Summer",
1: "Winter",
2: "Fall",
3: "Spring",
4: "Autumn"
}
'''
Tops = {
0: "T-shirt",
1: "Blazer",
2: "Sweater",
3: "Dress",
4: "Hoodie",
5: "Coat",
6: "Polo Shirt",
7: "Linen Shirt",
8: "Jacket",
9: "Cardigan",
10: "Tank Top",
11: "Jumpsuit",
12: "Long Sleeve",
13: "Blouse",
14: "Sweater"
}
Bottoms = {
0: "Shorts",
1: "Pants",
2: "Jeans",
3: "N/A",
4: "Joggers",
5: "Thermal",
6: "Chinos",
7: "Skirt",
8: "Trousers",
9: "Cargo Pants"
}
Top_Colors = {
0: "Light Blue",
1: "Navy",
2: "Maroon",
3: "Red",
4: "Gray",
5: "White",
6: "Green",
7: "Brown",
8: "Purple",
9: "Yellow",
10: "Orange",
11: "Teal"
}
Bottom_Colors = {
0: "Beige",
1: "Black",
2: "Dark Blue",
3: "White",
4: "Black",
5: "Navy",
6: "Green",
7: "Gray",
8: "Blue",
9: "Gold",
10: "Olive"
}
output_helper_list = [Tops, Bottoms, Top_Colors, Bottom_Colors]


data_inputs = {
    'Skin Tone': [0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 2, 0, 0, 1, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2, 2, 1],
    'Gender': [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'Temperature (°C)': [25, 10, 15, 30, 20, 5, 22, 28, 18, 12, 24, 32, 10, 20, 14, 17, 8, 23, 34, 16, 27, 9, 21, 26, 11, 22, 27, 15, 19, 30, 24, 18, 10, 25, 32, 28, 14, 21, 23, 17, 9, 26, 13, 20, 11],
    'Season': [0, 1, 2, 3, 4, 1, 0, 3, 2, 1, 0, 3, 4, 2, 1, 0, 4, 3, 1, 2, 4, 3, 0, 1, 3, 3, 4, 1, 0, 2, 1, 4, 3, 0, 1, 2, 4, 3, 0, 2, 3, 1, 2, 4, 3],
    'Occasion': [1, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 1, 0, 1, 2, 0, 1, 2, 1, 0, 1, 1, 0, 2, 1, 0, 2, 1, 1, 0, 2, 1, 0, 2, 1],
}


data_outputs = {
    'Top': [0, 1, 2, 6, 4, 5, 10, 7, 8, 9, 11, 12, 13, 6, 14, 0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 1, 3, 6, 2, 4, 7, 5, 1, 0, 3, 4, 2, 11, 12, 13, 9, 8, 14, 1, 10],
    'Bottom': [0, 1, 2, 1, 4, 5, 0, 6, 2, 7, 0, 3, 8, 9, 1, 0, 1, 2, 2, 4, 5, 6, 6, 7, 9, 1, 0, 2, 4, 5, 0, 1, 7, 6, 8, 9, 2, 3, 0, 1, 2, 3, 4, 5, 6],
    'Color Top': [5, 1, 2, 3, 4, 3, 5, 6, 7, 8, 9, 10, 11, 3, 1, 0, 3, 2, 4, 6, 9, 5, 1, 2, 3, 3, 4, 1, 2, 5, 6, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3],
    'Color Bottom': [0, 1, 2, 1, 1, 1, 5, 0, 2, 7, 0, 3, 2, 8, 10, 4, 7, 3, 1, 0, 5, 2, 5, 7, 8, 0, 1, 3, 2, 1, 0, 4, 3, 2, 5, 4, 1, 7, 6, 5, 3, 2, 1, 0, 4]
}


# Create the DataFrame
Parameters = pd.DataFrame(data_inputs)
Suggestions = pd.DataFrame(data_outputs)


X = Parameters
y = Suggestions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(40,55))

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Define a function to suggest an outfit
def suggest_outfit(parameters):
    # Convert the input parameters into a DataFrame with appropriate column names
    parameters_df = pd.DataFrame([parameters], columns=['Skin Tone', 'Gender', 'Temperature (°C)', 'Season', 'Occasion'])
    suggestion = knn.predict(parameters_df)
    return suggestion

def get_parameters():
    p = []
    p.append(int(input("What is your skin tone closest to? 0: Fair, 1: Medium, 2: Dark: ")))
    p.append(int(input("What is your Gender? 0: Male, 1: Female: ")))
    p.append(int(input("Temperature in C?: ")))
    p.append(int(input("What Season is it? 0: Summer, 1: Winter, 2: Fall, 3: Spring, 4: Autumn: ")))
    p.append(int(input("What type of occasion? 0: Formal, 1: Casual, 2: Semi-Formal: ")))
    return p

# Example usage
parameters_given = get_parameters()  # Example parameters



outfit_suggestion = suggest_outfit(parameters_given)
output = []
for i,j in zip(range(len(outfit_suggestion[0])), outfit_suggestion[0]):
    output.append(output_helper_list[i][j])

if output[1] == "N/A":
    output[3] = "N/A"
elif output[0] == "Dress" and parameters_given[1] == 0:
    output[0] = "Linen Shirt"
print("Outfit Suggestion:", output)