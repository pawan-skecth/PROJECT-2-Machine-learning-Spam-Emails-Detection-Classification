# PROJECT-2-Machine-learning-Spam-Emails-Detection-Classification


## 📌 **Introduction**  
Spam emails are **unwanted, irrelevant, or fraudulent messages** that clutter inboxes and pose security risks like **phishing, malware, and scams**. Detecting spam manually is impractical, so **Machine Learning (ML) models** are used to automate spam filtering.  

This project builds a **spam detection model** that classifies emails as **Spam or Not Spam (Ham)** based on text analysis and machine learning techniques.  

The project follows a **structured ML pipeline**:  
✅ **Data Collection** → ✅ **Exploratory Data Analysis (EDA)** → ✅ **Text Preprocessing** → ✅ **Feature Engineering** → ✅ **Model Training & Evaluation** → ✅ **Deployment using Gradio**  

---

## 🛠 **Steps Followed**  

---

## **📂 Step 1: Data Collection**  

### **Dataset Used:**  
- The dataset is sourced from **open-source spam datasets** containing labeled emails.  
- It consists of **two main columns**:  
  - `label` → **Spam (1) / Not Spam (0)**  
  - `message` → **Email text content**  

### **Sample Data:**  

| Label  | Message |
|--------|---------|
| Ham | "Hey, are we still meeting at 5 PM?" |
| Spam | "Congratulations! You've won a lottery. Click here to claim now!" |
| Ham | "Please find the attached report." |
| Spam | "URGENT! Your bank account will be suspended. Verify now!" |

---

## **📊 Step 2: Exploratory Data Analysis (EDA)**  

EDA helps understand **spam vs ham message distribution**, text patterns, and common words in spam emails.  

### **Key Insights from EDA:**  
1. **Spam emails contain frequent keywords**: "win", "free", "congratulations", "click", "urgent".  
2. **Spam messages are generally shorter** but contain **more exclamation marks** and **capitalized words**.  
3. **Ham emails contain more formal words** and personal references.  

---

## **🧹 Step 3: Text Preprocessing**  

Since emails are **text-based**, they must be **cleaned** before applying ML models.  

### **Preprocessing Steps:**  
✅ **Lowercasing:** Converts all text to lowercase.  
✅ **Removing Special Characters & Punctuation:** Cleans unnecessary symbols.  
✅ **Tokenization:** Splits sentences into words.  
✅ **Stopword Removal:** Removes common words (e.g., "the", "is", "and") that don’t affect classification.  
✅ **Lemmatization:** Converts words to their base forms (e.g., "running" → "run").  

---

## **📏 Step 4: Feature Engineering**  

Text data is **converted into numerical features** for ML models using **TF-IDF (Term Frequency-Inverse Document Frequency)**.  

### **Why TF-IDF?**  
✅ **Gives higher weight to important words** (e.g., "lottery" in spam emails).  
✅ **Reduces the impact of commonly used words**.  

**Example:**  
| Word | Frequency (TF) | Importance (IDF) | TF-IDF Score |
|------|--------------|----------------|--------------|
| Win  | High        | High           | High        |
| The  | High        | Low            | Low         |

---

## **🤖 Step 5: Model Training & Evaluation**  

We trained **multiple models** and compared their performance.  

### **Train-Test Split:**  
- **80% Training Data, 20% Test Data**  
- `X = Processed Email Text`, `y = Label (Spam=1, Ham=0)`.  

### **Models Used & Accuracy Scores:**  

| Model | Accuracy |
|---------|-----------|
| **Logistic Regression** | **98.2%** ✅ |
| Naïve Bayes | 97.8% |
| Random Forest | 96.5% |

🔹 **Logistic Regression performed the best!** 🎯  

---

## **📁 Step 6: Model Saving & Deployment**  

### **Saving Model using Pickle:**  
- The trained model was saved using **Pickle (`spam_model.pkl`)** for deployment.  

```python
import pickle
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### **Building Gradio Web App:**  
- Used **Gradio** to create a **simple UI**.  
- Users can **input an email**, and the model predicts whether it's **Spam or Not Spam**.  

```python
import gradio as gr

def predict_spam(email):
    input_text = vectorizer.transform([email])
    prediction = model.predict(input_text)[0]
    return "Spam" if prediction == 1 else "Not Spam"

iface = gr.Interface(fn=predict_spam, inputs="text", outputs="text")
iface.launch()
```

🔹 **Deployed locally using `iface.launch(share=True)`.**  

---

## **🚀 Results & Conclusion**  

### ✅ **Final Outcomes:**  
1. **Achieved high accuracy (98.2%) in spam detection.**  
2. **Effectively cleaned and preprocessed text data.**  
3. **Integrated a real-time spam detection system using Gradio.**  
4. **Model generalizes well for unseen emails.**  

### 📌 **Future Enhancements:**  
🚀 **Use advanced NLP models** like LSTM or Transformer-based models.  
🚀 **Deploy the model on a cloud platform** for real-world applications.  
🚀 **Train on larger datasets** to improve spam detection accuracy.  

🔹 **GitHub Repository**: https://github.com/pawan-skecth/PROJECT-2-Machine-learning-Spam-Emails-Detection-Classification/edit/main/README.md 

---

## 🎯 **Final Thoughts**  
This project **successfully applies Machine Learning to real-world spam detection**. The **Gradio web app** makes it easy for users to **paste an email and instantly check if it's spam or not**.  

📌 **Fast, accurate, and deployable!** 🚀  

-
