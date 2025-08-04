from typing import Dict, Any

def process_sequence(sequence: str) -> Dict[str, Any]:
    """Process a biological sequence and return analysis results."""
    length = len(sequence)
    gc_content = (sequence.count('G') + sequence.count('C')) / length * 100 if length > 0 else 0
    at_content = (sequence.count('A') + sequence.count('T')) / length * 100 if length > 0 else 0
    
    return {
        "length": length,
        "gc_content": gc_content,
        "at_content": at_content,
        "sequence_preview": sequence[:50] + ('...' if length > 50 else '')
    }

def generate_summary(sequence: str) -> Dict[str, Any]:
    """Generate a summary of the analysis for the given sequence."""
    analysis_results = process_sequence(sequence)
    
    summary = {
        "metadata": {
            "sequence_length": analysis_results["length"],
            "gc_content": analysis_results["gc_content"],
            "at_content": analysis_results["at_content"],
        },
        "sequence_preview": analysis_results["sequence_preview"]
    }
    
    return summary

def analyze_sequences(sequences: list) -> Dict[str, Any]:
    """Analyze a list of sequences and return a comprehensive report."""
    results = []
    for seq in sequences:
        results.append(generate_summary(seq))
    
    return {
        "total_sequences": len(sequences),
        "analysis_results": results
    }