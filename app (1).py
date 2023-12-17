import streamlit as st
import pandas as pd

from clothingRS import show_recommendation

import streamlit as st
# import your_recommendation_module  # Import the module where your recommendation code is defined

# Set up the Streamlit app title and description
st.title("Recommendation App")
st.write('You can upload an image to get recommendation based on your image! Pre-trained DenseNet121 model will help you!')

mapping_df = pd.read_csv('fashion-dataset/fashion-dataset/images.csv')
# Upload an image using Streamlit's built-in file uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    file_name = uploaded_image.name
    st.write(f"File Name: {file_name}")
    index = int(mapping_df[mapping_df['filename'] == file_name].index.tolist()[0])
    print(index)
    print(type(index))
    # index = int(file_name.split('.')[0])
    st.write(index)
    # Add a button to trigger the recommendation generation
    if st.button("Generate Recommendations"):
        # Perform recommendation generation based on the uploaded image
        recommendation_list = show_recommendation(index)
        print(recommendation_list)
        # Display the recommendations to the user
        st.subheader("Recommendations:")
        for recommendation in recommendation_list:
            path = f'fashion-dataset/fashion-dataset/images/{recommendation}'
            # index_list = mapping_df[mapping_df['filename'] == recommendation].index.tolist()
            # index = int(index_list[0])
            # index = index_list[0]
            st.image(path)
# [number.jpg] 가 아니라 인덱스로 불러와야되네
# You can define the 'generate_recommendations' function in your recommendation module
# It should take the uploaded image as input and return a list of recommendations
