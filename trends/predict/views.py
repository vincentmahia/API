from rest_framework.response import Response
from rest_framework.decorators import api_view
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@api_view(["POST"])
def predict_view(request):
    file = request.FILES.get("file")
    target = request.POST.get("target")

    if not file or not target:
        return Response({"error": "Please upload a file and specify the target column."}, status=400)

    try:
        # Read CSV or Excel file
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return Response({"error": "Invalid file format. Upload CSV or Excel."}, status=400)

        # Ensure target column exists
        df.columns = df.columns.str.strip()
        if target not in df.columns:
            return Response({"error": f"Target column '{target}' not found."}, status=400)

        def cleaning(dataset):
            num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
            cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
            preprocessing = make_column_transformer(
                (num_pipeline, make_column_selector(dtype_include=np.number)),
                (cat_pipeline, make_column_selector(dtype_include=object))
            )
            preprocessing.fit_transform(dataset)
            return preprocessing
        cleaning(df)
        # Involve only numeric features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numerical_features:
            numerical_features.remove(target)  # Remove target variable

        # Functions to generate attribute combination
        def generate_strong_correlation_features(df, target, features,num_features,threshold=0.5, prefix="corr_"):
            """Generates ratio-based, multiplication-based, and transformed features for highly correlated features with the target."""
            
            # Compute correlation with the target variable
            correlation = df[features].corrwith(df[target]).abs()
            
            # Select features with strong correlation
            strong_corr_features = correlation[correlation > threshold].index.tolist()
            
            if not strong_corr_features:
                print("No strongly correlated features found. Try lowering the threshold.")
                return pd.DataFrame()
            
            new_features = {}
            count = 0

            # Generate ratio-based and multiplication-based features
            for i in range(len(strong_corr_features)):
                for j in range(i + 1, len(strong_corr_features)):
                    if count >= num_features // 3:  # About 1/3 of new features from ratios
                        break
                    ratio_name = f"{prefix}{strong_corr_features[i]}_div_{strong_corr_features[j]}"
                    new_features[ratio_name] = df[strong_corr_features[i]] / (df[strong_corr_features[j]] + 1e-5)  # Avoid div by zero
                    count += 1
                    
                    if count >= num_features // 3:  # About 1/3 from multiplications
                        break
                    mult_name = f"{prefix}{strong_corr_features[i]}_mul_{strong_corr_features[j]}"
                    new_features[mult_name] = df[strong_corr_features[i]] * df[strong_corr_features[j]]
                    count += 1

            # Apply transformations (log, sqrt, exp)
            for feature in strong_corr_features:
                if count >= num_features:
                    break
                new_features[f"{prefix}log_{feature}"] = np.log1p(df[feature])  # log(1 + x) to handle zero values
                count += 1
                if count >= num_features:
                    break
                new_features[f"{prefix}sqrt_{feature}"] = np.sqrt(df[feature])
                count += 1
                if count >= num_features:
                    break
                new_features[f"{prefix}exp_{feature}"] = np.exp(df[feature] / df[feature].max())  # Normalize before exp
                count += 1

            # Convert to DataFrame and merge with original
            new_features_df = pd.DataFrame(new_features, index=df.index)
            
            return new_features_df

        # Apply feature engineering based on strong correlation with price
        new_features = generate_strong_correlation_features(df, target=target, features=numerical_features, threshold=0.5, num_features=6)

        # Merge with original dataset
        df_combined = pd.concat([df, new_features], axis=1)
        cleaning(df_combined)
        df_combined = df_combined.select_dtypes(include=np.number)

          # Prepare Data
        X = df_combined.drop(columns=[target])
        y = df_combined[target]
        X = pd.get_dummies(X, drop_first=True)
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train XGBoost Model
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
                # Hyperparameter Grid Search
        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of boosting rounds
            'max_depth': [3, 5, 7],  # Maximum depth of trees
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'subsample': [0.8, 1],  # Fraction of samples used per boosting round
            'colsample_bytree': [0.8, 1]  # Fraction of features used per boosting round
        }

        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Best Model from Grid Search
        best_model = grid_search.best_estimator_

        # Predict & Evaluate
        y_pred = best_model.predict(X_test)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        r2 = round(r2_score(y_test, y_pred), 5)*100

        # Predict & Evaluate
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = round(r2_score(y_test, y_pred), 5)*100

        return Response({
            "rmse": rmse,
            "r2_score": f'{r2}%',
            "message": "Prediction successful!"
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)