import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

class Dataset:
    def __init__(self, data : pd.DataFrame = None, data_path = None, train = True):
        self.data = data
        self.train = train
        self.encoders = []
        if data_path is not None:
            self.load_dataset(data_path)
        
    def load_dataset(self, data_path):
        self.data = pd.read_csv(data_path)
        
    def preprocess(self):
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.preprocess_strings()
        # self.normalize_data()
        if self.train:
            target_col = self.data.pop("Salary")
            self.data["Salary"] = target_col
            self.features = self.data.iloc[:, :-1]
            self.target = self.data.iloc[:, -1]
        else:
            self.features = self.data
        
    def preprocess_strings(self):
        
        def deprecate_of(title):
            if ' of ' in title:
                parts = title.split(' of ')
                return f"{parts[1].strip()} {parts[0].strip()}"
            return title
        
        self.data['Job Title'] = self.data['Job Title'].str.lower()
        self.data['Job Title'].replace("human resources", "hr", inplace=True)
        self.data['Job Title'] = self.data['Job Title'].apply(deprecate_of)
        self.data['Job Title'].replace("representative", "rep", inplace=True)
        self.data['Job Title'].replace(["senior", "junior"], "", inplace=True)
        self.data['Gender'] = self.data['Gender'].str.lower()
        self.data['Education Level'] = self.data['Education Level'].str.lower() 
        if self.train:
            self.encode_strings()
    
    def encode_strings(self, encoders = None):
        features = ['Gender', 'Education Level', 'Job Title']
        
        if encoders is None:
            for feature in features:
                # unknown = {feature : ["Unknown"]}
                # unknown = pd.DataFrame(unknown)
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                enc.fit(self.data[[feature]])
                # enc.fit(pd.concat([self.data[feature], unknown]))
                self.encoders.append(enc)
        encs = encoders if encoders is not None else self.encoders    
        for i, feature in enumerate(features):
            enc = encs[i]
            self.data[feature] = enc.transform(self.data[[feature]])
        
        # self.data['Gender'].replace({'male' : 0, 'female' : 1}, inplace=True)       
        # self.data = pd.get_dummies(self.data, columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)
        # bool_cols = self.data.select_dtypes(include='bool').columns
        # self.data[bool_cols] = self.data[bool_cols].astype(int)
        
    def normalize_data(self):
        columns = ['Age', 'Years of Experience', 'Salary']
        for column in columns:
            if column in self.data.columns:
                offset, scale = self.get_scaling_params('Age')
                self.data['Age'] = (self.data['Age'] - offset) * scale
                offset, scale = self.get_scaling_params('Years of Experience')
                self.data['Years of Experience'] = (self.data['Years of Experience'] - offset) * scale
    
    def get_scaling_params(self, column=None):
        arr = self.data[column]
        return np.min(arr), 1 / (np.max(arr) - np.min(arr))
    
    def get_encoder(self):
        return self.data.iloc[:, 2:-1]
    
def normalize_data(df):
        columns = ['Age', 'Years of Experience', 'Salary']
        for column in columns:
            if column in df.columns:
                offset, scale = df.get_scaling_params('Age')
                df['Age'] = (df['Age'] - offset) * scale
                offset, scale = df.get_scaling_params('Years of Experience')
                df['Years of Experience'] = (df['Years of Experience'] - offset) * scale
    
if __name__ == '__main__':
    path = "D:/python/mathModelling/hw4/Salary Data.csv"
    dataset = Dataset(data_path=path)
    dataset.preprocess()
    model = LinearRegression()
    train_x, test_x, train_y, test_y = train_test_split(dataset.features, dataset.target, test_size=0.5, random_state=0)
    
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    df = pd.DataFrame({'gt' : test_y, 'pred' : y_pred})
    print(df)
    mse = root_mean_squared_error(test_y, y_pred)
    
    r2 = r2_score(test_y, y_pred)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Coefficient of Determination (R²): {r2:.2f}')
    
    new_data = {
    'Age': [32],
    'Gender': ['Male'],
    'Education Level': ["Bachelor's"],
    'Job Title': ['HR Project Manager'],
    'Years of Experience': [5]
    }
    new_data = pd.DataFrame(new_data)
    new_data = Dataset(data=new_data, train=False)
    new_data.preprocess()
    new_data.encode_strings(dataset.encoders)
    
    new_X = new_data.data
    predicted_salary = model.predict(new_X)[0]
    print(f'Предсказанная зарплата: {predicted_salary:.2f}')