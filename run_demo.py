"""Simple runner script for the Intellumina knowledge graph demo.

Usage:
    python run_demo.py

This will build the graph for a sample text, visualize it (if matplotlib available),
export files and run the evaluation using an embedded ground truth.
"""
from intellumina import KnowledgeGraphBuilder, KnowledgeGraphEvaluator

EXAMPLE_TEXT = """
Dr. Sarah Chen is a creative and ambitious marine biologist who works at
the Ocean Research Institute. She collaborated with Dr. James Liu, a
meticulous and organized oceanographer from Stanford University. Sarah
is very outgoing and enthusiastic when presenting her research, while
James is more introverted and quiet but highly reliable.

Their colleague, Maria Rodriguez, is a kind and compassionate data scientist
who joined the team last year. Maria is calm and stable under pressure,
which complements Sarah's energetic personality. The team works in
San Francisco and frequently visits the Monterey Bay research station.

Sarah mentored three graduate students and led an innovative coral reef
restoration project. James designed the experimental protocols with his
typical disciplined approach.
"""

GROUND_TRUTH = {
    'entities': [
        'Sarah Chen', 'James Liu', 'Maria Rodriguez',
        'Ocean Research Institute', 'Stanford University',
        'San Francisco', 'Monterey Bay'
    ],
    'relationships': [
        ('Sarah Chen', 'works', 'Ocean Research Institute'),
        ('Sarah Chen', 'collaborated', 'James Liu'),
        ('James Liu', 'works', 'Stanford University'),
    ],
    'personalities': {
        'Sarah Chen': {
            'Openness': 0.85,
            'Conscientiousness': 0.75,
            'Extraversion': 0.90,
            'Agreeableness': 0.70,
            'Neuroticism': 0.30
        },
        'James Liu': {
            'Openness': 0.60,
            'Conscientiousness': 0.90,
            'Extraversion': 0.25,
            'Agreeableness': 0.75,
            'Neuroticism': 0.35
        },
        'Maria Rodriguez': {
            'Openness': 0.70,
            'Conscientiousness': 0.80,
            'Extraversion': 0.60,
            'Agreeableness': 0.90,
            'Neuroticism': 0.25
        }
    }
}


def main():
    print("[STEP 1] Building knowledge graph...")
    kg = KnowledgeGraphBuilder()
    kg.extract_entities_and_relations(EXAMPLE_TEXT)
    kg.extract_personality_traits(EXAMPLE_TEXT)
    kg.print_summary()

    print("[STEP 2] Visualizing graph (if available)...")
    try:
        kg.visualize_graph()
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("[STEP 3] Exporting graph files...")
    kg.export_to_json()
    kg.export_to_graphml()

    print("[STEP 4] Running evaluation...")
    evaluator = KnowledgeGraphEvaluator(kg, GROUND_TRUTH)
    results = evaluator.run_full_evaluation(verbose=True)
    evaluator.export_evaluation_report()

    print("\n✓ Demo complete. Check the generated JSON/GraphML files.")


if __name__ == '__main__':
    main()
