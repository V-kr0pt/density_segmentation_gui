def save_processing_results():
    """Save processing results to JSON file"""
    try:
        results_file = os.path.join(os.getcwd(), 'output', 'sam2_processing_results.json')
        
        # Prepare JSON-serializable results
        json_results = {
            filename: {
                "status": result["status"],
                "message": result["message"],
                "has_results": result.get("results") is not None
            }
            for filename, result in st.session_state["sam2_completed_files"].items()
        }
        
        with open(results_file, 'w') as f:
            import json
            json.dump(json_results, f, indent=2)
        
        st.success(f"Results saved to {results_file}")
        
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
