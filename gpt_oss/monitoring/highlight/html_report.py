"""
HTML report generator for hallucination monitoring results.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from jinja2 import Template


class HTMLReportGenerator:
    """Generator for HTML hallucination monitoring reports."""
    
    def __init__(self):
        """Initialize the HTML report generator."""
        self.template = self._get_template()
    
    def _get_template(self) -> Template:
        """Get the HTML template for reports."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-OSS Hallucination Monitor Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .timestamp {
            opacity: 0.8;
            margin-top: 10px;
        }
        .content {
            padding: 30px;
        }
        .summary {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .risk-score {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .risk-low { color: #28a745; }
        .risk-medium { color: #ffc107; }
        .risk-high { color: #dc3545; }
        .signals {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .signal-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .signal-score {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .signal-name {
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        .completion-section {
            margin: 30px 0;
        }
        .completion-text {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            line-height: 1.8;
        }
        .highlighted-text {
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
        }
        .highlight-green { background-color: #d4edda; color: #155724; }
        .highlight-yellow { background-color: #fff3cd; color: #856404; }
        .highlight-red { background-color: #f8d7da; color: #721c24; }
        .highlight-purple { background-color: #e2d9f3; color: #5a2d82; }
        .highlight-blue { background-color: #d1ecf1; color: #0c5460; }
        .details-section {
            margin: 30px 0;
        }
        .collapsible {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin: 10px 0;
            overflow: hidden;
        }
        .collapsible-header {
            background: #e9ecef;
            padding: 15px 20px;
            cursor: pointer;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .collapsible-header:hover {
            background: #dee2e6;
        }
        .collapsible-content {
            padding: 20px;
            display: none;
        }
        .collapsible-content.active {
            display: block;
        }
        .toggle-icon {
            transition: transform 0.3s;
        }
        .toggle-icon.rotated {
            transform: rotate(180deg);
        }
        .config-section {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .config-item {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .config-label {
            font-weight: 600;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .config-value {
            margin-top: 5px;
            font-family: 'Courier New', monospace;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }
        @media (max-width: 768px) {
            .signals {
                grid-template-columns: 1fr;
            }
            .config-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 GPT-OSS Hallucination Monitor</h1>
            <div class="timestamp">{{ timestamp }}</div>
        </div>
        
        <div class="content">
            <div class="summary">
                <h2>Risk Assessment Summary</h2>
                <div class="risk-score risk-{{ risk_level }}">{{ risk_score }}</div>
                <div style="text-align: center; font-size: 1.2em; color: #666;">
                    Risk Level: <strong>{{ risk_level.upper() }}</strong>
                </div>
                
                <div class="signals">
                    {% for signal_name, signal_data in signals.items() %}
                    <div class="signal-card">
                        <div class="signal-name">{{ signal_name.replace('_', ' ').title() }}</div>
                        <div class="signal-score">{{ "%.3f"|format(signal_data.score) }}</div>
                        <div style="color: #666; font-size: 0.9em;">{{ signal_data.description }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="completion-section">
                <h2>Analyzed Completion</h2>
                <div class="completion-text">{{ highlighted_completion }}</div>
            </div>
            
            <div class="details-section">
                <h2>Detailed Analysis</h2>
                
                {% for detector_name, detector_data in detector_details.items() %}
                <div class="collapsible">
                    <div class="collapsible-header" onclick="toggleSection('{{ detector_name }}')">
                        <span>{{ detector_name.replace('_', ' ').title() }} Analysis</span>
                        <span class="toggle-icon" id="icon-{{ detector_name }}">▼</span>
                    </div>
                    <div class="collapsible-content" id="content-{{ detector_name }}">
                        <pre>{{ detector_data | tojson(indent=2) }}</pre>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="config-section">
                <h2>Configuration</h2>
                <div class="config-grid">
                    {% for key, value in config.items() %}
                    <div class="config-item">
                        <div class="config-label">{{ key.replace('_', ' ').title() }}</div>
                        <div class="config-value">{{ value }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by GPT-OSS Hallucination Monitor v{{ version }}</p>
            <p>For more information, visit the <a href="https://github.com/openai/gpt-oss">GPT-OSS repository</a></p>
        </div>
    </div>
    
    <script>
        function toggleSection(sectionId) {
            const content = document.getElementById('content-' + sectionId);
            const icon = document.getElementById('icon-' + sectionId);
            
            content.classList.toggle('active');
            icon.classList.toggle('rotated');
        }
        
        // Auto-expand high-risk sections
        document.addEventListener('DOMContentLoaded', function() {
            const riskLevel = '{{ risk_level }}';
            if (riskLevel === 'high') {
                // Auto-expand all sections for high-risk cases
                document.querySelectorAll('.collapsible-content').forEach(content => {
                    content.classList.add('active');
                });
                document.querySelectorAll('.toggle-icon').forEach(icon => {
                    icon.classList.add('rotated');
                });
            }
        });
    </script>
</body>
</html>
        """
        return Template(template_str)
    
    def _highlight_text(self, text: str, spans: List[Dict[str, Any]]) -> str:
        """Highlight text with spans using HTML."""
        if not spans:
            return text
        
        # Sort spans by start position (descending) to avoid index issues
        sorted_spans = sorted(spans, key=lambda x: x['start'], reverse=True)
        
        highlighted_text = text
        
        for span in sorted_spans:
            start = span['start']
            end = span['end']
            color = span.get('color', 'blue')
            reason = span.get('reason', '')
            
            # Create highlight span
            highlight_html = f'<span class="highlighted-text highlight-{color}" title="{reason}">{highlighted_text[start:end]}</span>'
            
            # Replace the text
            highlighted_text = highlighted_text[:start] + highlight_html + highlighted_text[end:]
        
        return highlighted_text
    
    def _get_signal_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each signal type."""
        return {
            'self_consistency': 'Semantic agreement across multiple generations',
            'nli_faithfulness': 'Entailment against prompt and context',
            'numeric_sanity': 'Arithmetic and unit consistency',
            'retrieval_support': 'Support from provided context documents',
            'jailbreak_heuristics': 'Safety risk assessment'
        }
    
    def generate_report(self, results: Dict[str, Any], config: Dict[str, Any], 
                       version: str = "0.1.0") -> str:
        """Generate HTML report from monitoring results."""
        
        # Prepare data for template
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_score = f"{results['risk_score']:.3f}"
        risk_level = results['risk_level']
        
        # Prepare signals data
        signals = {}
        signal_descriptions = self._get_signal_descriptions()
        
        for signal_name, score in results['signals'].items():
            signals[signal_name] = {
                'score': score,
                'description': signal_descriptions.get(signal_name, 'Signal score')
            }
        
        # Highlight completion text
        highlighted_completion = self._highlight_text(
            results.get('completion', ''),
            results.get('spans', [])
        )
        
        # Prepare detector details
        detector_details = {}
        for detector_name, details in results.get('detector_details', {}).items():
            detector_details[detector_name] = details
        
        # Render template
        html_content = self.template.render(
            timestamp=timestamp,
            risk_score=risk_score,
            risk_level=risk_level,
            signals=signals,
            highlighted_completion=highlighted_completion,
            detector_details=detector_details,
            config=config,
            version=version
        )
        
        return html_content
    
    def save_report(self, results: Dict[str, Any], config: Dict[str, Any], 
                   output_path: str, version: str = "0.1.0") -> str:
        """Generate and save HTML report to file."""
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate HTML content
        html_content = self.generate_report(results, config, version)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
