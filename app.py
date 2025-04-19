from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

def analyze_intent(agreement_type, important_info, extra_info):
    analysis_result = ""

    if not agreement_type:
        analysis_result += "Please specify the type of legal agreement you need (e.g., NDA, Lease Agreement, Employment Contract).\n"
        return analysis_result

    analysis_result += f"Analyzing request for a '{agreement_type}' agreement.\n"

    if not important_info:
        analysis_result += "Please provide the key details relevant to this agreement, such as the parties involved, the subject matter, and the duration or key terms.\n"
    else:
        analysis_result += f"Key details provided: '{important_info}'.\n"

    if extra_info:
        analysis_result += f"Additional details or specific clauses mentioned: '{extra_info}'.\n"
    else:
        analysis_result += "Consider adding any specific clauses or customization requirements you have for this agreement.\n"

    analysis_result += "\nTo generate a comprehensive and accurate agreement, please ensure you provide sufficient details."
    return analysis_result

def generate_agreement(agreement_type, important_info, extra_info):
    analysis = analyze_intent(agreement_type, important_info, extra_info)
    if "Please specify the type of legal agreement" in analysis or "Please provide the key details" in analysis:
        return analysis

    # --- Basic Agreement Generation Logic (Illustrative) ---
    if agreement_type.lower() == "nda":
        return f"""Generated Non-Disclosure Agreement based on the provided information:
        - General Details: {important_info}
        - Specific Clauses: {extra_info}
        This is a preliminary draft and requires thorough review and customization."""
    elif agreement_type.lower() == "lease agreement":
        return f"""Generated Lease Agreement based on the provided information:
        - Key Terms: {important_info}
        - Additional Clauses: {extra_info}
        This is a preliminary draft and requires thorough review and customization."""
    elif agreement_type.lower() == "employment agreement":
        return f"""Generated Employment Agreement based on the provided information:
        - Core Details: {important_info}
        - Further Stipulations: {extra_info}
        This is a preliminary draft and requires thorough review and customization."""
    elif agreement_type.lower() == "franchise agreement":
        return f"""Generated Franchise Agreement based on the provided information:
        - Basic Information: {important_info}
        - Specific Terms: {extra_info}
        This is a preliminary draft and requires thorough review and customization."""
    elif agreement_type.lower() == "contractor agreement":
        return f"""Generated Contractor Agreement based on the provided information:
        - Project Overview: {important_info}
        - Additional Stipulations: {extra_info}
        This is a preliminary draft and requires thorough review and customization."""
    elif agreement_type.lower() == "rent agreement":
        return f"""Generated Rent Agreement based on the provided information:
        - Property and Parties: {important_info}
        - Rental Terms: {extra_info}
        This is a preliminary draft and requires thorough review and customization."""
    else:
        return f"Agreement type '{agreement_type}' is not currently supported for full generation. Analysis: {analysis}"

@app.route('/generate-agreement', methods=['POST'])
def handle_generation():
    data = request.get_json()
    agreement_type = data.get('agreement_type', '')
    important_info = data.get('important_info', '')
    extra_info = data.get('extra_info', '')

    response_text = generate_agreement(agreement_type, important_info, extra_info)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)