"""Knowledge graph builder and evaluator module extracted from notebook.

This module provides two main classes:
- KnowledgeGraphBuilder: extracts entities, relations and personality traits and builds a NetworkX graph.
- KnowledgeGraphEvaluator: runs quantitative & qualitative metrics against an optional ground truth.

The spaCy model is loaded lazily to avoid heavy imports at module import time.
"""
from collections import defaultdict
import json
import math
from typing import Dict

import networkx as nx
import numpy as np

# Lazy spaCy import / loader to avoid loading model at import time
def get_nlp():
    try:
        import spacy
    except ImportError as e:
        raise ImportError("spaCy is required. Install with: pip install spacy") from e

    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("Could not load 'en_core_web_sm'. Run: python -m spacy download en_core_web_sm") from e

# Personality adjective to Big Five mapping (kept small and explicit)
PERSONALITY_TRAITS = {
    # Openness to Experience
    'creative': ('Openness', 0.8),
    'curious': ('Openness', 0.7),
    'imaginative': ('Openness', 0.8),
    'artistic': ('Openness', 0.9),
    'adventurous': ('Openness', 0.7),
    'innovative': ('Openness', 0.8),

    # Conscientiousness
    'organized': ('Conscientiousness', 0.8),
    'responsible': ('Conscientiousness', 0.8),
    'disciplined': ('Conscientiousness', 0.9),
    'meticulous': ('Conscientiousness', 0.9),
    'reliable': ('Conscientiousness', 0.8),
    'punctual': ('Conscientiousness', 0.7),
    'ambitious': ('Conscientiousness', 0.7),

    # Extraversion
    'outgoing': ('Extraversion', 0.8),
    'sociable': ('Extraversion', 0.8),
    'energetic': ('Extraversion', 0.7),
    'talkative': ('Extraversion', 0.8),
    'enthusiastic': ('Extraversion', 0.8),
    'introverted': ('Extraversion', -0.8),
    'shy': ('Extraversion', -0.7),
    'quiet': ('Extraversion', -0.6),

    # Agreeableness
    'kind': ('Agreeableness', 0.8),
    'cooperative': ('Agreeableness', 0.8),
    'compassionate': ('Agreeableness', 0.9),
    'friendly': ('Agreeableness', 0.7),
    'empathetic': ('Agreeableness', 0.9),
    'generous': ('Agreeableness', 0.8),
    'trusting': ('Agreeableness', 0.7),

    # Neuroticism
    'anxious': ('Neuroticism', 0.8),
    'nervous': ('Neuroticism', 0.7),
    'worried': ('Neuroticism', 0.7),
    'calm': ('Neuroticism', -0.8),
    'stable': ('Neuroticism', -0.8),
    'confident': ('Neuroticism', -0.7),
    'resilient': ('Neuroticism', -0.8),
}


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from text using spaCy and NetworkX."""

    def __init__(self, nlp=None):
        # Accept an already loaded spaCy nlp pipeline or load lazily
        self.nlp = nlp
        self.graph = nx.DiGraph()
        self.personality_scores = defaultdict(lambda: defaultdict(list))

    def _ensure_nlp(self):
        if self.nlp is None:
            self.nlp = get_nlp()

    def extract_entities_and_relations(self, text: str):
        """Extract named entities and simple subject-verb-object relations.

        Returns list of nodes in the graph after extraction.
        """
        self._ensure_nlp()
        doc = self.nlp(text)

        # Add named entities of interest
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                self.graph.add_node(ent.text, type=ent.label_)

        # Very small relation extractor based on dependency labels
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ['nsubj', 'nsubjpass'] and token.head.pos_ == 'VERB':
                    subject = token.text
                    verb = token.head.text
                    for child in token.head.children:
                        if child.dep_ in ['dobj', 'pobj', 'attr']:
                            obj = child.text
                            if subject in self.graph.nodes and obj in self.graph.nodes:
                                self.graph.add_edge(subject, obj, relation=verb)
        return list(self.graph.nodes)

    def extract_personality_traits(self, text: str):
        """Find adjectives in the text and map to Big Five traits using PERSONALITY_TRAITS."""
        self._ensure_nlp()
        doc = self.nlp(text)
        for sent in doc.sents:
            person_entities = [ent.text for ent in sent.ents if ent.label_ == 'PERSON']
            for token in sent:
                if token.pos_ == 'ADJ':
                    adj_lower = token.lemma_.lower()
                    if adj_lower in PERSONALITY_TRAITS:
                        trait, score = PERSONALITY_TRAITS[adj_lower]
                        associated_person = person_entities[0] if person_entities else None
                        if associated_person:
                            self.personality_scores[associated_person][trait].append(score)
                            trait_node = f"{trait}_{adj_lower}"
                            self.graph.add_node(trait_node, type='Trait', trait_category=trait, adjective=adj_lower)
                            self.graph.add_edge(associated_person, trait_node, relation='exhibits', score=score)

    def aggregate_personality_scores(self) -> Dict[str, Dict[str, float]]:
        """Aggregate collected adjective scores into normalized [0,1] Big Five scores."""
        aggregated = {}
        for person, traits in self.personality_scores.items():
            aggregated[person] = {}
            for trait, scores in traits.items():
                avg_score = sum(scores) / len(scores)
                # Original mappings used roughly -1..1; normalize to 0..1 for presentation
                normalized = (avg_score + 1) / 2
                aggregated[person][trait] = round(normalized, 2)
            if person in self.graph.nodes:
                self.graph.nodes[person]['personality'] = aggregated[person]
        return aggregated

    def visualize_graph(self, figsize=(14, 10)):
        """Visualize the graph using matplotlib and networkx."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise RuntimeError("matplotlib is required to visualize the graph")

        plt.figure(figsize=figsize)
        color_map = {
            'PERSON': '#FF6B6B',
            'ORG': '#4ECDC4',
            'GPE': '#45B7D1',
            'LOC': '#96CEB4',
            'Trait': '#FFEAA7'
        }
        node_colors = [
            color_map.get(self.graph.nodes[n].get('type', 'OTHER'), '#DFE6E9')
            for n in self.graph.nodes()
        ]
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors,
                node_size=2000, font_weight='bold', font_size=9)
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=7)
        plt.title("Knowledge Graph with Personality Traits", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def export_to_json(self, filename='knowledge_graph.json'):
        data = {'nodes': [], 'edges': []}
        for node in self.graph.nodes():
            node_data = {'id': node}
            node_data.update(self.graph.nodes[node])
            data['nodes'].append(node_data)
        for u, v, attrs in self.graph.edges(data=True):
            edge_data = {'source': u, 'target': v}
            edge_data.update(attrs)
            data['edges'].append(edge_data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Graph exported to {filename}")

    def export_to_graphml(self, filename='knowledge_graph.graphml'):
        G_clean = nx.DiGraph()
        for n, attrs in self.graph.nodes(data=True):
            clean_attrs = {k: (json.dumps(v) if isinstance(v, (dict, list, set, tuple)) else v)
                           for k, v in attrs.items()}
            G_clean.add_node(n, **clean_attrs)
        for u, v, attrs in self.graph.edges(data=True):
            clean_attrs = {k: (json.dumps(v) if isinstance(v, (dict, list, set, tuple)) else v)
                           for k, v in attrs.items()}
            G_clean.add_edge(u, v, **clean_attrs)
        nx.write_graphml(G_clean, filename)
        print(f"✓ Graph exported to {filename}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("=" * 60)
        print(f"\nTotal Nodes: {self.graph.number_of_nodes()}")
        print(f"Total Edges: {self.graph.number_of_edges()}")
        node_types = defaultdict(list)
        for node in self.graph.nodes():
            node_types[self.graph.nodes[node].get('type', 'OTHER')].append(node)
        print("\n--- NODES BY TYPE ---")
        for t, nodes in node_types.items():
            print(f"{t}: {nodes}")
        print("\n--- RELATIONSHIPS ---")
        for u, v, data in self.graph.edges(data=True):
            print(f"{u} --[{data.get('relation', 'related_to')}]--> {v}")
        print("\n--- PERSONALITY PROFILES (Big Five) ---")
        profiles = self.aggregate_personality_scores()
        for person, scores in profiles.items():
            print(f"\n{person}:")
            for trait, score in scores.items():
                print(f"  {trait}: {score}")


class KnowledgeGraphEvaluator:
    """Evaluation suite for the Knowledge Graph."""

    def __init__(self, kg_builder: KnowledgeGraphBuilder, ground_truth: Dict = None):
        self.kg = kg_builder
        self.graph = kg_builder.graph
        self.ground_truth = ground_truth or {}
        self.evaluation_results = {}

    def calculate_entity_metrics(self) -> Dict:
        if 'entities' not in self.ground_truth:
            return {'status': 'No ground truth provided'}

        extracted_entities = set(n for n in self.graph.nodes()
                                if self.graph.nodes[n].get('type') in ['PERSON', 'ORG', 'GPE', 'LOC'])
        true_entities = set(self.ground_truth['entities'])

        true_positives = len(extracted_entities & true_entities)
        false_positives = len(extracted_entities - true_entities)
        false_negatives = len(true_entities - extracted_entities)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'extracted_count': len(extracted_entities),
            'true_count': len(true_entities)
        }

    def calculate_relationship_metrics(self) -> Dict:
        if 'relationships' not in self.ground_truth:
            return {'status': 'No ground truth provided'}

        extracted_rels = set()
        for u, v, data in self.graph.edges(data=True):
            if self.graph.nodes[u].get('type') != 'Trait' and self.graph.nodes[v].get('type') != 'Trait':
                extracted_rels.add((u, v, data.get('relation', 'unknown')))

        true_rels = set(tuple(r) for r in self.ground_truth['relationships'])

        exact_matches = len(extracted_rels & true_rels)

        extracted_pairs = set((u, v) for u, v, _ in extracted_rels)
        true_pairs = set((u, v) for u, v, _ in true_rels)
        partial_matches = len(extracted_pairs & true_pairs)

        precision = exact_matches / len(extracted_rels) if extracted_rels else 0
        recall = exact_matches / len(true_rels) if true_rels else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'extracted_count': len(extracted_rels),
            'true_count': len(true_rels)
        }

    def calculate_personality_metrics(self) -> Dict:
        if 'personalities' not in self.ground_truth:
            return {'status': 'No ground truth provided'}

        personality_profiles = self.kg.aggregate_personality_scores()
        true_profiles = self.ground_truth['personalities']

        errors = []
        for person in true_profiles:
            if person in personality_profiles:
                for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
                    if trait in true_profiles[person] and trait in personality_profiles[person]:
                        error = abs(true_profiles[person][trait] - personality_profiles[person][trait])
                        errors.append(error)

        mae = float(np.mean(errors)) if errors else 0.0

        people_with_traits = len(personality_profiles)
        total_people = len([n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'PERSON'])
        coverage = people_with_traits / total_people if total_people > 0 else 0

        complete_profiles = sum(1 for p in personality_profiles.values() if len(p) == 5)
        completeness = complete_profiles / people_with_traits if people_with_traits > 0 else 0

        return {
            'mean_absolute_error': round(mae, 3),
            'trait_coverage': round(coverage, 3),
            'profile_completeness': round(completeness, 3),
            'people_with_traits': people_with_traits,
            'complete_profiles': complete_profiles,
            'target_mae': '< 0.20 (Good)',
            'target_coverage': '> 0.80 (Good)'
        }

    def calculate_graph_structure_metrics(self) -> Dict:
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0

        undirected = self.graph.to_undirected()
        num_components = nx.number_connected_components(undirected)

        degrees = [d for _, d in self.graph.degree()]
        avg_degree = float(np.mean(degrees)) if degrees else 0

        isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]

        clustering = nx.average_clustering(undirected) if num_nodes > 0 else 0

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': round(density, 4),
            'avg_degree': round(avg_degree, 2),
            'connected_components': num_components,
            'isolated_nodes': len(isolated),
            'clustering_coefficient': round(clustering, 3),
            'isolated_node_list': isolated[:5]
        }

    def assess_semantic_coherence(self) -> Dict:
        relationships = []
        for u, v, data in self.graph.edges(data=True):
            if self.graph.nodes[u].get('type') != 'Trait' and self.graph.nodes[v].get('type') != 'Trait':
                relationships.append({
                    'subject': u,
                    'relation': data.get('relation', 'unknown'),
                    'object': v,
                    'coherence_score': None
                })

        return {
            'total_relationships': len(relationships),
            'sample_relationships': relationships[:10],
            'instructions': 'Manually score each relationship 1-5 for plausibility',
            'target_avg_score': '> 4.0'
        }

    def check_personality_interpretability(self) -> Dict:
        personality_profiles = self.kg.aggregate_personality_scores()

        interpretability_report = []
        for person, traits in personality_profiles.items():
            evidence = []
            for neighbor in self.graph.neighbors(person):
                if self.graph.nodes[neighbor].get('type') == 'Trait':
                    adjective = self.graph.nodes[neighbor].get('adjective')
                    trait_cat = self.graph.nodes[neighbor].get('trait_category')
                    evidence.append(f"{adjective} → {trait_cat}")

            interpretability_report.append({
                'person': person,
                'traits': traits,
                'evidence': evidence,
                'evidence_count': len(evidence)
            })

        return {
            'profiles': interpretability_report,
            'avg_evidence_per_person': round(np.mean([p['evidence_count'] for p in interpretability_report]), 2) if interpretability_report else 0
        }

    def detect_hallucinations(self) -> Dict:
        isolated = [n for n in self.graph.nodes()
                   if self.graph.degree(n) == 0 and self.graph.nodes[n].get('type') != 'Trait']

        unusual_relations = []
        for u, v, data in self.graph.edges(data=True):
            u_type = self.graph.nodes[u].get('type')
            v_type = self.graph.nodes[v].get('type')
            if u_type in ['ORG', 'GPE', 'LOC'] and v_type == 'Trait':
                unusual_relations.append((u, data.get('relation'), v))

        return {
            'isolated_entities': isolated,
            'unusual_relationships': unusual_relations,
            'hallucination_rate': len(unusual_relations) / self.graph.number_of_edges() if self.graph.number_of_edges() > 0 else 0,
        }

    def run_full_evaluation(self, verbose=True) -> Dict:
        print("\n" + "=" * 70)
        print("COMPREHENSIVE KNOWLEDGE GRAPH EVALUATION")
        print("=" * 70)

        entity_metrics = self.calculate_entity_metrics()
        self.evaluation_results['entity_metrics'] = entity_metrics
        if verbose:
            for k, v in entity_metrics.items():
                print(f"  {k}: {v}")

        rel_metrics = self.calculate_relationship_metrics()
        self.evaluation_results['relationship_metrics'] = rel_metrics
        if verbose:
            for k, v in rel_metrics.items():
                print(f"  {k}: {v}")

        personality_metrics = self.calculate_personality_metrics()
        self.evaluation_results['personality_metrics'] = personality_metrics
        if verbose:
            for k, v in personality_metrics.items():
                print(f"  {k}: {v}")

        structure_metrics = self.calculate_graph_structure_metrics()
        self.evaluation_results['structure_metrics'] = structure_metrics
        if verbose:
            for k, v in structure_metrics.items():
                if k != 'isolated_node_list':
                    print(f"  {k}: {v}")

        coherence = self.assess_semantic_coherence()
        self.evaluation_results['coherence'] = coherence
        print(f"  Total relationships to assess: {coherence['total_relationships']}")

        interpretability = self.check_personality_interpretability()
        self.evaluation_results['interpretability'] = interpretability
        print(f"  Avg evidence per person: {interpretability['avg_evidence_per_person']}")

        hallucinations = self.detect_hallucinations()
        self.evaluation_results['hallucinations'] = hallucinations
        print(f"  Isolated entities: {len(hallucinations['isolated_entities'])}")
        print(f"  Unusual relationships: {len(hallucinations['unusual_relationships'])}")

        # Composite score (simple weighted average)
        score_components = []
        weights = []

        if 'f1_score' in entity_metrics and entity_metrics.get('f1_score') != 'No ground truth provided':
            score_components.append(entity_metrics['f1_score'])
            weights.append(0.25)

        if 'f1_score' in rel_metrics and rel_metrics.get('f1_score') != 'No ground truth provided':
            score_components.append(rel_metrics['f1_score'])
            weights.append(0.25)

        if 'mean_absolute_error' in personality_metrics:
            mae_score = max(0, 1 - personality_metrics['mean_absolute_error'])
            score_components.append(mae_score)
            weights.append(0.20)

        if 'trait_coverage' in personality_metrics:
            score_components.append(personality_metrics['trait_coverage'])
            weights.append(0.15)

        score_components.append(structure_metrics['clustering_coefficient'])
        weights.append(0.15)

        if score_components:
            overall_score = sum(s * w for s, w in zip(score_components, weights)) / sum(weights)
            print(f"\n  Overall Quality Score: {overall_score:.3f} / 1.000")

        return self.evaluation_results

    def export_evaluation_report(self, filename='evaluation_report.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Evaluation report exported to {filename}")
