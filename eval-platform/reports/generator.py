from fpdf import FPDF
from datetime import datetime
from pathlib import Path

def generate_report(results: list[dict], model: str, category: str, run_id: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "EVAL REPORT", ln=True)
    
    # Model info
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Model: {model}", ln=True)
    pdf.cell(0, 8, f"Category: {category}", ln=True)
    pdf.cell(0, 8, f"Run ID: {run_id}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    
    # Scores
    pdf.ln(5)
    pdf.set_font("Helvetica", style="B", size=13)
    pdf.cell(0, 10, "SCORES", ln=True)
    pdf.set_font("Helvetica", size=12)
    avg_ensemble = sum(r["confidence"] for r in results) / len(results)
    pdf.cell(0, 8, f"Avg Confidence: {avg_ensemble:.2f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", style="B", size=13)
    pdf.cell(0, 10, "RESULTS", ln=True)
    pdf.set_font("Helvetica", size=12)
    
    for result in results:
              pdf.multi_cell(0, 10, f"Category: {result['category']}", ln=True)
              pdf.multi_cell(0, 10, f"Prompt ID: {result['id']}", ln=True)
              pdf.multi_cell(0, 10, f"Prompt Text: {result['prompt_text']}", ln=True)
              pdf.multi_cell(0, 10, f"Expected Behaviour: {result['expected_behaviour']}", ln=True)
              pdf.multi_cell(0, 10, f"Severity: {result['severity']}", ln=True)
              pdf.multi_cell(0, 10, f"Response: {result['response']}", ln=True)
              pdf.multi_cell(0, 10, f"Confidence: {result['confidence']}", ln=True)
              pdf.multi_cell(0, 10, f"Timestamp: {result['timestamp']}", ln=True)
              pdf.ln()

    out_path = Path(__file__).parent / f"report_{run_id[:8]}.pdf"
    pdf.output(str(out_path))
    return str(out_path)