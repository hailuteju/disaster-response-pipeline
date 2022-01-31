# Disaster Response Pipeline Project

### Project summary

In this project, we will analyze thousands of real messages - provided by Figure Eight -
that were sent during natural disasters either via social media or directly to 
disaster response organizations.

We will build an ETL pipeline that processes message and category data from CSV
files and load them into a SQLite database which then will be read by an ML pipeline
to create and save a multi-output supervised learning model. Then the web app will
extract data from the database to provide data visualization and uses the model
to classify new messages for 36 categories.

ML is critical to helping different organizations understand which messages are 
relevant to them and which messages to prioritize. During natural disasters is
when they have the least capacity to filter out messages that matter and find 
basic messages such as using keyword searches to provide trivial results. This 
project aims to provide such a tool.


### Files in the repository

```yaml
* app    
|-- template  
    |-- master.html   # main page of web app       
    |-- go.html  # classification result page of web app       
|-- run.py  # Flask file that runs app      

* data     
|-- disaster_categories.csv  # data to process     
|-- disaster_messages.csv  # data to process         
|-- process_data.py   # ETL pipeline that processes and saves the cleaned data to a database       
|-- DisasterResponse.db  # database to save clean data to     

* models    
|-- train_classifier.py  # ML pipeline that trains a multioutput supervised model and saves the model       
|-- model.pkl  # saved model    

* README.md 

```


### Instructions:
1. Run the following commands in the project's root directory to install the dependencies in the `requirements.txt` file and set up the database and model.
    - To install the packages in the `requirements.txt` file 
    
        ```python
        pip install -r requirements.txt
      
        ```

    - To run ETL pipeline that cleans data and stores in database
   
        ```python
        python data/process_data.py \
        data/disaster_messages.csv \
        data/disaster_categories.csv \
        data/DisasterResponse.db
        
        ``` 
        
    - To run ML pipeline that trains classifier and saves
    
        ```python
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        
        ```

2. Run the following command in the app's directory to run the web app.

    ```python 
    python app/run.py
   
    ```

5. Go to http://localhost:3001/ and enter a message to classify in the textbox.
