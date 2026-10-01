import os
from datetime import datetime
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_report(summary_stats: dict, output_path: str = "reports/Sentiment_Analysis_Report.pdf") -> str:
    """
    Generates a professional PDF report summarizing sentiment analysis results using ReportLab.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E293B'),
        alignment=0,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15
    )

    heading2_style = ParagraphStyle(
        'DocHeading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("TweetSense – Sentiment Analysis Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Author: TweetSense System", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3B82F6'), spaceAfter=15))

    # Executive Summary Section
    elements.append(Paragraph("1. Executive Summary", heading2_style))
    total = summary_stats.get("total_tweets", 0)
    pos = summary_stats.get("positive_count", 0)
    neg = summary_stats.get("negative_count", 0)
    neu = summary_stats.get("neutral_count", 0)

    pos_pct = f"{(pos / total * 100):.1f}%" if total > 0 else "0.0%"
    neg_pct = f"{(neg / total * 100):.1f}%" if total > 0 else "0.0%"
    neu_pct = f"{(neu / total * 100):.1f}%" if total > 0 else "0.0%"

    elements.append(Paragraph(
        f"This automated report presents sentiment classification metrics for <b>{total}</b> analyzed tweets. "
        f"The distribution shows <b>{pos}</b> ({pos_pct}) Positive tweets, <b>{neg}</b> ({neg_pct}) Negative tweets, "
        f"and <b>{neu}</b> ({neu_pct}) Neutral tweets.",
        body_style
    ))
    elements.append(Spacer(1, 10))

    # Summary Statistics Table
    table_data = [
        ["Sentiment Metric", "Count", "Percentage"],
        ["Positive Tweets", str(pos), pos_pct],
        ["Negative Tweets", str(neg), neg_pct],
        ["Neutral Tweets", str(neu), neu_pct],
        ["Total Analyzed", str(total), "100.0%"]
    ]

    t = Table(table_data, colWidths=[200, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 15))

    # Model Performance Section
    elements.append(Paragraph("2. Machine Learning Model Performance", heading2_style))
    model_name = summary_stats.get("model_name", "Logistic Regression")
    acc = summary_stats.get("accuracy", "N/A")
    prec = summary_stats.get("precision", "N/A")
    rec = summary_stats.get("recall", "N/A")
    f1 = summary_stats.get("f1_score", "N/A")

    elements.append(Paragraph(f"Primary Classification Algorithm: <b>{model_name}</b>", body_style))
    
    perf_data = [
        ["Metric", "Score"],
        ["Accuracy", str(acc)],
        ["Precision (Weighted)", str(prec)],
        ["Recall (Weighted)", str(rec)],
        ["F1-Score (Weighted)", str(f1)]
    ]

    t_perf = Table(perf_data, colWidths=[250, 250])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    elements.append(t_perf)
    elements.append(Spacer(1, 15))

    # Key Methodology & Limitations Section
    elements.append(Paragraph("3. Academic Methodology & Limitations", heading2_style))
    elements.append(Paragraph(
        "• <b>Feature Extraction:</b> TF-IDF (Term Frequency - Inverse Document Frequency) with unigrams and bigrams.<br/>"
        "• <b>Preprocessing:</b> Lowercasing, URL/@mention removal, hashtag preservation, contraction expansion, emoji mapping, stopword filtering with negation preservation.<br/>"
        "• <b>Academic Disclaimer:</b> Model outputs represent statistical confidence derived from training patterns. Sarcasm, contextual nuances, and domain-specific slang may impact individual tweet classifications.",
        body_style
    ))

    # Build Document
    doc.build(elements)
    return output_path
