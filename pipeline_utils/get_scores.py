import numpy as np

def get_scores(subject_data):

    # Read in neurocognitive test scores
    if input("Does the subject have test scores? (y/n): ") == 'y':

        score_names = ['MMSE Total Score', 'GDSCALE Total Score', 'FAQ Total Score', 'NPI-Q Total Score']

        def get_score_input(prompt, min_val, max_val):

            while True:

                try:

                    value = input(prompt)

                    if value == "":

                        return np.nan
                    
                    value = float(value)

                    if min_val <= value <= max_val:

                        return value
                    
                    else:

                        print(f"Value must be between {min_val} and {max_val}.")

                except ValueError:

                    print("Invalid input. Please enter a number or press enter for NaN.")

        mmse = get_score_input("Enter 'MMSE Total Score' (0-30), or press enter if no score: ", 0, 30)
        gdscale = get_score_input("Enter 'GDSCALE Total Score' (0-15), or press enter if no score: ", 0, 15)
        faq = get_score_input("Enter 'FAQ Total Score' (0-30), or press enter if no score: ", 0, 30)
        npiq = get_score_input("Enter 'NPI-Q Total Score' (0-12), or press enter if no score: ", 0, 12)
        
        subject_data['scores'] = [mmse, gdscale, faq, npiq]
        
        subject_data['score_names'] = score_names

    else:

        subject_data['scores'], subject_data['score_names'] = None, None

    return subject_data