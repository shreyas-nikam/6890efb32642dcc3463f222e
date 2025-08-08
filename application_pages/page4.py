
import streamlit as st
import pickle
import os

def run_page4():
    st.header("4. Model Persistence")
    st.markdown("You can download the best-performing fitted ARIMAX model for future use.")

    if 'fitted_model' in st.session_state and st.session_state['fitted_model'] is not None:
        best_model = st.session_state['fitted_model']
        
        # Create a dummy directory for saving locally in Streamlit environment
        output_dir = './models_temp/'
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, 'macro_pd_arimax_taiwan.pkl')

        try:
            with open(model_filename, 'wb') as file:
                pickle.dump(best_model, file)
            
            with open(model_filename, 'rb') as file:
                st.download_button(
                    label="Download Fitted ARIMAX Model (.pkl)",
                    data=file,
                    file_name="macro_pd_arimax_taiwan.pkl",
                    mime="application/octet-stream"
                )
            st.success(f"Model ready for download.")
            st.info(f"Model details: Type - {best_model.__class__.__name__}, AIC - {best_model.aic:.2f}")

        except Exception as e:
            st.error(f"Error preparing model for download: {e}")
    else:
        st.warning("No model has been successfully fitted yet to save.")

if __name__ == "__main__":
    run_page4()
