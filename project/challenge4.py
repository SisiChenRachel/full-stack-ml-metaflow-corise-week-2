from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)

from metaflow.cards import Table, Markdown, Artifact, Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# move your labeling function from earlier in the notebook here
labeling_function = lambda row: 1 if row['rating'] >= 4 else 0


class NLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))
        
        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        self.df_original= _has_review_df
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.review_tfidf_vectorizer)


    @step
    def review_tfidf_vectorizer(self):


        from sklearn.feature_extraction.text import TfidfVectorizer

        # Text Vectorization (TF-IDF Vectorizer)
        tfidf_vectorizer = TfidfVectorizer(max_features=1000,stop_words='english') 
        self.X_train = self.traindf['review']
        self.y_train = self.traindf['label']
        self.X_val = self.valdf['review']
        self.y_val = self.valdf['label']
        self.X_train_tfidf = tfidf_vectorizer.fit_transform(self.X_train)
        self.X_val_tfidf = tfidf_vectorizer.transform(self.X_val)
        self.next(self.hyperparameter_tuning)


    @step
    def hyperparameter_tuning(self):
        param_grid = {
                'C': [1.0, 10.0],
                'gamma': ['scale', 'auto']
            }
        grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=3)
        grid_search.fit(self.X_train_tfidf, self.y_train)
        results = grid_search.cv_results_

        # Display the results as a pandas DataFrame (optional)
        self.df_hp = pd.DataFrame(results)
        self.next(self.end)

    @card(
        type="corise"
    )    

    @step
    def end(self):
        print(self.df_hp)
        
        df_result = self.df_hp
        heatmap_data = df_result.pivot(index='param_C', columns='param_gamma', values='mean_test_score')


        # Create a heatmap
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis')
        plt.title('Hyperparameter Grid Search Results')
        plt.xlabel('Gamma')
        plt.ylabel('C')
        current.card.append(Image.from_matplotlib(fig))
       
        df_cloud = self.df_original
        text = ' '.join(df_cloud['review_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig = plt.figure(figsize=(20, 12))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud from Text Column')
        current.card.append(Image.from_matplotlib(fig))

if __name__ == "__main__":
   NLPFlow()
