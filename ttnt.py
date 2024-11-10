import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Đọc dữ liệu từ file CSV
data = pd.read_csv("D:/Project/questions.csv", encoding='utf-8')
questions = data['questions']  # Lấy cột 'question'
answers = data['answers']      # Lấy cột 'answer'

# Chuyển đổi văn bản thành dạng số bằng TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
y = answers

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Hàm dự đoán cho câu hỏi mới (từ khóa)
def ask_question(question):
    # Chuyển câu hỏi thành vector
    question_vec = vectorizer.transform([question])
    
    # Dự đoán với mô hình Naive Bayes
    answer = model.predict(question_vec)
    
    return answer[0]

def search_related_questions(keyword):
    # Tìm các câu hỏi có chứa từ khóa
    related_questions = [q for q in questions if keyword.lower() in q.lower()]
    
    # Nếu có câu hỏi liên quan, trả về câu trả lời của câu hỏi đầu tiên liên quan
    if related_questions:
        # Dự đoán câu trả lời cho câu hỏi liên quan đầu tiên
        related_question = related_questions[0]
        return ask_question(related_question)
    else:
        return "Không tìm thấy câu hỏi phù hợp."

# Tạo giao diện Tkinter
def on_submit():
    question = entry.get()  # Lấy câu hỏi từ người dùng
    if question.strip() == "":  # Kiểm tra nếu câu hỏi trống
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập câu hỏi.")
    else:
        # Kiểm tra nếu câu hỏi là từ khóa
        if any(keyword in question.lower() for keyword in questions.str.lower()):
            answer = search_related_questions(question)  # Tìm kiếm câu hỏi liên quan
        else:
            answer = ask_question(question)  # Dự đoán câu trả lời
        result_label.config(text=f"Câu trả lời: {answer}")  # Hiển thị câu trả lời
        
# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Hỏi Đáp AI")

# Tạo widget
label = tk.Label(root, text="Nhập câu hỏi hoặc từ khóa của bạn:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

submit_button = tk.Button(root, text="Hỏi", command=on_submit)
submit_button.pack(pady=10)

result_label = tk.Label(root, text="Câu trả lời sẽ hiển thị ở đây", wraplength=400)
result_label.pack(pady=20)

# Chạy giao diện
root.mainloop()