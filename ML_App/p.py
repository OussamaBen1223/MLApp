import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import PhotoImage

class MLApp():
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning App")
        self.root.geometry("800x600")

        self.data = None
        self.target_column = None
        self.best_model = None
        self.scaler = None
        self.new_data_entries = []
        self.model_scores = {}
        self.encoder = LabelEncoder()

        # Configure the style
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12), padding=10)
        style.configure('TLabel', font=('Helvetica', 12), padding=10)
        style.configure('TNotebook.Tab', font=('Helvetica', 12, 'bold'), padding=10)
        
        style.map("TButton",
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '!disabled', 'blue'), ('active', 'blue')])

        # Create a canvas
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(fill='both', expand=True)

        # Create notebook for tabs on the canvas
        self.notebook = ttk.Notebook(self.root)
        self.canvas.create_window(0, 0, anchor='nw', window=self.notebook, width=800, height=600)

        # Tab for main application
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text='Main')

        # Tab for model scores
        self.scores_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.scores_tab, text='Model Scores')

        # Configure UI elements in main tab
        self.label = ttk.Label(self.main_tab, text="Machine Learning Application", font=("Helvetica", 16, 'bold'))
        self.label.pack(pady=10)

        self.load_button = ttk.Button(self.main_tab, text="Télécharger Dataset", command=self.load_data)
        self.load_button.pack(pady=5)

        # Label to display the name of the loaded dataset file
        self.dataset_label = ttk.Label(self.main_tab, text="", font=("Helvetica", 10))
        self.dataset_label.pack(pady=5)

        # Buttons that will be hidden initially
        self.show_data_button = ttk.Button(self.main_tab, text="Afficher les Données", command=self.show_data)
        self.show_data_button.pack(pady=5)
        self.show_data_button.pack_forget()  # Hide button initially

        self.stats_button = ttk.Button(self.main_tab, text="Afficher les Statistiques", command=self.display_data_statistics)
        self.stats_button.pack(pady=5)
        self.stats_button.pack_forget()  # Hide button initially

        self.visualize_button = ttk.Button(self.main_tab, text="Visualiser les données", command=self.visualize_data)
        self.visualize_button.pack(pady=5)
        self.visualize_button.pack_forget()  # Hide button initially

        self.delete_column_button = ttk.Button(self.main_tab, text="Supprimer des Colonnes", command=self.delete_columns)
        self.delete_column_button.pack(pady=5)
        self.delete_column_button.pack_forget()  # Hide button initially

        self.target_label = ttk.Label(self.main_tab, text="Changer la colonne cible")
        self.target_label.pack(pady=5)
        self.target_label.pack_forget()  # Hide label initially

        self.target_combobox = ttk.Combobox(self.main_tab)
        self.target_combobox.pack(pady=5)
        self.target_combobox.pack_forget()  # Hide combobox initially

        self.preprocess_button = ttk.Button(self.main_tab, text="Prétraiter les données", command=self.do_preprocessing)
        self.preprocess_button.pack(pady=5)
        self.preprocess_button.pack_forget()  # Hide button initially

        self.train_button = ttk.Button(self.main_tab, text="Entraîner et Évaluer", command=self.do_training)
        self.train_button.pack(pady=5)
        self.train_button.pack_forget()  # Hide button initially

        self.new_data_label = ttk.Label(self.main_tab, text="Prédire une nouvelle entrée")
        self.new_data_label.pack(pady=5)
        self.new_data_label.pack_forget()  # Hide label initially

        self.predict_button = ttk.Button(self.main_tab, text="Entrer une nouvelle donnée", command=self.create_new_data_entries)
        self.predict_button.pack(pady=5)
        self.predict_button.pack_forget()

        self.edit_button = ttk.Button(self.main_tab, text="Modifier les valeurs NaN", command=self.open_edit_window)
        self.edit_button.pack(pady=5)
        self.edit_button.pack_forget() # Hide button initially
        
    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Données chargées avec succès")
            print(self.data.head())  # Display the first few rows of the dataset

            # Update the combobox with column names
            self.target_combobox['values'] = self.data.columns.tolist()
            self.target_combobox.current(0)  # Set the default selected item

            # Display the dataset file name
            self.dataset_label.config(text=f"Fichier chargé: {file_path.split('/')[-1]}")

            # Show hidden buttons after data is loaded
            self.show_data_button.pack(pady=5)
            self.stats_button.pack(pady=5)
            self.visualize_button.pack(pady=5)
            self.edit_button.pack()
            self.delete_column_button.pack(pady=5)
            self.target_label.pack(pady=5)
            self.target_combobox.pack(pady=5)
            self.preprocess_button.pack(pady=5)
            self.train_button.pack(pady=5)
            self.new_data_label.pack(pady=5)
            self.predict_button.pack(pady=5)

    def show_data(self):
        if self.data is None:
            messagebox.showerror("Erreur", "Aucune donnée disponible.")
            return

        # Create a new window to display data
        data_window = tk.Toplevel(self.root)
        data_window.title("Données Importées")

        # Frame for centering
        frame = ttk.Frame(data_window, padding=(10, 10, 10, 10))
        frame.pack(expand=True, fill="both")

        # Add a scrollable text box to display data
        text_area = tk.Text(frame, wrap="none", height=30, width=100, padx=10, pady=10)
        text_area.pack(expand=True, fill="both")

        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=text_area.yview)
        scroll_y.pack(side="right", fill="y")

        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=text_area.xview)
        scroll_x.pack(side="bottom", fill="x")

        text_area.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        # Insert data into the text box
        data_text = self.data.head(10000).to_string()
        text_area.insert("end", data_text)

        # Disable text editing
        text_area.config(state="disabled")

    def visualize_data(self):
        if self.data is None:
            messagebox.showerror("Erreur", "Aucune donnée disponible pour visualisation.")
            return

        # Create a new window for data visualization
        visualize_window = tk.Toplevel(self.root)
        visualize_window.title("Visualisation des données")

        # Add buttons for different visualizations
        correlation_button = ttk.Button(visualize_window, text="Heatmap de Corrélation", command=self.show_correlation_heatmap)
        correlation_button.pack(pady=5)

        pairplot_button = ttk.Button(visualize_window, text="Pairplot", command=self.show_pairplot)
        pairplot_button.pack(pady=5)

        hist_button = ttk.Button(visualize_window, text="Histogrammes", command=self.show_histograms)
        hist_button.pack(pady=5)

    def show_correlation_heatmap(self):
        # Create and display a correlation heatmap
        if self.data is not None:
            # Encode categorical columns
            encoder = LabelEncoder()
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.data[col] = encoder.fit_transform(self.data[col])

            # Calculate correlation matrix
            correlation_matrix = self.data.corr()

            # Plot heatmap
            plt.figure(figsize=(15, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.show()
        else:
            messagebox.showwarning("Warning", "Veuillez d'abord charger les données")

    def show_pairplot(self):
        # Create and display a pairplot
        plt.figure(figsize=(20, 10))
        sns.pairplot(self.data)
        plt.show()

    def show_histograms(self):
        # Create and display histograms for each column
        self.data.hist(bins=40, figsize=(15, 10))
        plt.suptitle('Histogrammes des Variables')
        plt.show()

    def open_edit_window(self):
        self.edit_window = tk.Toplevel(self.root)
        self.edit_window.title("Modifier les valeurs manquants")
        self.edit_window.geometry("400x300")

        tk.Label(self.edit_window, text="Selectioner une colonne:").pack(pady=5)
        columns = self.data.columns.tolist()
        self.column_combobox = ttk.Combobox(self.edit_window, values=columns)
        self.column_combobox.pack(pady=5)

        tk.Label(self.edit_window, text="Selectioner la strategy:").pack(pady=5)
        self.strategy_combobox = ttk.Combobox(self.edit_window, values=["most_frequent", "constant"])
        self.strategy_combobox.pack(pady=5)
        self.strategy_combobox.bind("<<ComboboxSelected>>", self.on_strategy_selected)

        self.constant_value_entry = tk.Entry(self.edit_window)
        self.constant_value_entry.pack(pady=5)
        self.constant_value_entry.pack_forget()

        self.save_button = ttk.Button(self.edit_window, text="Enregistrer", command=self.save_imputation)
        self.save_button.pack(pady=5)

    def on_strategy_selected(self, event):
        strategy = self.strategy_combobox.get()
        if strategy == "constant":
            self.constant_value_entry.pack(pady=5, after=self.strategy_combobox)
            self.save_button.pack(pady=5, after=self.constant_value_entry)
        else:
            self.constant_value_entry.pack_forget()
            self.save_button.pack(pady=5, after=self.strategy_combobox)

    def save_imputation(self):
        column_name = self.column_combobox.get()
        strategy = self.strategy_combobox.get()
        if strategy == "most_frequent":
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            self.data[[column_name]] = imputer.fit_transform(self.data[[column_name]])
        elif strategy == "constant":
            constant_value = self.constant_value_entry.get()
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=float(constant_value))
            self.data[[column_name]] = imputer.fit_transform(self.data[[column_name]])
        print(self.data)  # Print the dataframe to verify the changes
        # Show confirmation window
        messagebox.showinfo("Confirmation", f"les Valeurs NaN dans la colonne '{column_name}' sont remplacés par la stratégie'{strategy}'.")
        # Close the edit window
        self.edit_window.destroy()

    def do_preprocessing(self):
        if self.data is not None:
            for col in self.data.select_dtypes(include=['object']).columns:
                self.data[col] = self.data[col].astype('category').cat.codes

            self.data.fillna(self.data.mean(), inplace=True)
            messagebox.showinfo("Info", "Prétraitement réalisé avec succès")
        else:
            messagebox.showwarning("Warning", "Veuillez d'abord charger les données")

    def display_data_statistics(self):
        if self.data is None:
            messagebox.showerror("Erreur", "Aucune donnée disponible.")
            return

        # Display all basic statistics
        stats_text = "Statistiques sur le jeu de données :\n"
        stats_text += f"Nombre de lignes : {len(self.data)}\n"
        stats_text += f"Nombre de colonnes : {len(self.data.columns)}\n"
        stats_text += "\nNoms des colonnes :\n" + ", ".join(self.data.columns) + "\n"
        stats_text += f"\nTypes de données :\n{self.data.dtypes}\n"
        stats_text += f"\nRésumé statistique :\n{self.data.describe(include='all')}"

        # Display statistics in a dialog with a scrollable text box
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Statistiques du Dataset")

        # Frame for centering
        frame = ttk.Frame(stats_window, padding=(10, 10, 10, 10))
        frame.pack(expand=True, fill="both")

        # Add a scrollable text box to display statistics
        text_area = tk.Text(frame, wrap="word", height=30, width=100, padx=10, pady=10)
        text_area.pack(expand=True, fill="both")

        scroll_bar = ttk.Scrollbar(frame, command=text_area.yview)
        scroll_bar.pack(side="right", fill="y")

        text_area.config(yscrollcommand=scroll_bar.set)

        # Insert statistics text
        text_area.insert("end", stats_text)

        # Center the text
        text_area.tag_configure("center", justify='center')
        text_area.tag_add("center", 1.0, "end")

        # Disable text editing
        text_area.config(state="disabled")

    def do_training(self):
        if self.data is not None:
            self.target_column = self.target_combobox.get()
            if self.target_column is not None:
                X = self.data.drop(columns=[self.target_column])
                y = self.data[self.target_column]

                # Determine if the problem is classification or regression
                if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 20:
                    problem_type = "regression"
                else:
                    problem_type = "classification"

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Apply StandardScaler
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

                # Define model parameters for GridSearchCV
                if problem_type == "classification":
                    models = {
                        "K-Nearest Neighbors" : KNeighborsClassifier(),
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest": RandomForestClassifier(),
                        "Support Vector Machine": SVC(),
                        "Decision Tree": DecisionTreeClassifier()
                    }

                    param_grids = {
                        "K-Nearest Neighbors" : {"n_neighbors": [3, 5, 11, 19]},
                        "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
                        "Random Forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]},
                        "Support Vector Machine": {"C": [0.01, 0.1, 1, 10], "kernel": ["linear", "rbf"]},
                        "Decision Tree": {"max_depth": [None, 10, 20]}
                    }
                else:
                    models = {
                        "SGD Regressor": SGDRegressor(),
                        "Random Forest": RandomForestRegressor(),
                        "Support Vector Machine": SVR(),
                        "Decision Tree": DecisionTreeRegressor()
                    }

                    param_grids = {
                        "SGD Regressor": {"alpha": [0.0001, 0.001, 0.01, 0.1], "loss": ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']},
                        "Random Forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]},
                        "Support Vector Machine": {"C": [0.01, 0.1, 1, 10], "kernel": ["linear", "rbf"]},
                        "Decision Tree": {"max_depth": [None, 10, 20]}
                    }

                best_score = float('-inf')
                for model_name in models:
                    model = models[model_name]
                    param_grid = param_grids[model_name]
                    grid_search = GridSearchCV(model, param_grid, cv=5)
                    grid_search.fit(X_train, y_train)
                    score = grid_search.best_score_

                    # Store model scores
                    self.model_scores[model_name] = score

                    if score > best_score:
                        best_score = score
                        self.best_model = grid_search.best_estimator_

                self.display_model_scores()

                if problem_type == "classification":
                    y_pred = self.best_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"Meilleur model: {self.best_model}")
                    print(f"Accuracy: {accuracy}")
                    messagebox.showinfo("Info", f"Meilleur model: {self.best_model}\nAccuracy: {accuracy}")
                else:
                    y_pred = self.best_model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    print(f"Meilleur model: {self.best_model}")
                    print(f"Mean Squared Error: {mse}")
                    messagebox.showinfo("Info", f"Meilleur model: {self.best_model}\nMean Squared Error: {mse}")
            else:
                messagebox.showwarning("Warning", "Veuillez sélectionner la colonne cible")
        else:
            messagebox.showwarning("Warning", "Veuillez d'abord charger les données")

    def display_model_scores(self):
        for widget in self.scores_tab.winfo_children():
            widget.destroy()

        scores_text = "\n".join([f"{model}: {score:.4f}" for model, score in self.model_scores.items()])
        scores_label = ttk.Label(self.scores_tab, text=scores_text, padding=(10, 10, 10, 10))
        scores_label.pack(pady=10)

    def create_new_data_entries(self):
        # Create a new window for new data entry
        self.new_data_window = tk.Toplevel(self.root)
        self.new_data_window.title("Enterer une nouvelle donnée")

        # Create a frame with a scrollbar
        frame = ttk.Frame(self.new_data_window, padding=(10, 10, 10, 10))
        frame.pack(expand=True, fill="both")

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Clear previous entries
        self.new_data_entries = []
        for col in self.data.drop(columns=[self.target_column]).columns:
            label = ttk.Label(scrollable_frame, text=f"Saisir une valeur pour {col}:")
            label.pack(pady=5)
            entry = ttk.Entry(scrollable_frame)
            entry.pack(pady=5)
            self.new_data_entries.append(entry)

        # Add a button to predict the new entry
        predict_button = ttk.Button(scrollable_frame, text="Predict", command=self.predict_new_entry)
        predict_button.pack(pady=10)

    def predict_new_entry(self):
        if self.best_model is not None:
            try:
                new_data = [float(entry.get()) for entry in self.new_data_entries]
                new_data = np.array(new_data).reshape(1, -1)
                new_data_scaled = self.scaler.transform(new_data)
                prediction = self.best_model.predict(new_data_scaled)
                print("Prediction:", prediction)
                messagebox.showinfo("la Prediction" , f"Prediction de {self.target_column}: {prediction}")
            except ValueError:
                messagebox.showerror("Error", "Veuillez saisir des valeurs numériques valides pour tous les champs.")
        else:
            messagebox.showwarning("Warning", "Veuillez d'abord entraîner le modèle")

    def delete_columns(self):
        if self.data is None:
            messagebox.showerror("Erreur", "Aucune donnée disponible.")
            return

        # Create a new window for column deletion
        self.delete_window = tk.Toplevel(self.root)
        self.delete_window.title("Supprimer des Colonnes")

        # Create a frame with a scrollbar
        frame = ttk.Frame(self.delete_window, padding=(10, 10, 10, 10))
        frame.pack(expand=True, fill="both")

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Create checkboxes for each column
        self.column_vars = {}
        for col in self.data.columns:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(scrollable_frame, text=col, variable=var)
            chk.pack(pady=5)
            self.column_vars[col] = var

        # Add a button to confirm column deletion
        confirm_button = ttk.Button(scrollable_frame, text="Confirmer", command=self.confirm_delete_columns)
        confirm_button.pack(pady=10)

    def confirm_delete_columns(self):
        columns_to_delete = [col for col, var in self.column_vars.items() if var.get()]

        if columns_to_delete:
            self.data.drop(columns=columns_to_delete, inplace=True)
            messagebox.showinfo("Info", f"Colonnes supprimées: {', '.join(columns_to_delete)}")
            self.target_combobox['values'] = self.data.columns.tolist()  # Update the combobox with updated columns
        else:
            messagebox.showinfo("Info", "Aucune colonne sélectionnée pour suppression")

        self.delete_window.destroy()  # Close the deletion window

# Main application execution
if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
