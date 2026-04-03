"""
NEST 2026 ULTIMATE PREDICTION SYSTEM - ELITE VERSION v3.2
==========================================================
Adapted from GATE Prediction System with full logic preservation.
Fixed version with better data validation and subject handling.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# ML imports with graceful degradation
try:
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                                   StackingRegressor, HistGradientBoostingRegressor)
    from sklearn.linear_model import BayesianRidge, Ridge, HuberRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False

# ==========================================
# TOPIC NORMALIZATION MAP
# ==========================================
# Fixes topic fragmentation by mapping duplicate variants to canonical names
# Adapted for NEST subjects: Physics, Chemistry, Biology, Mathematics, General

TOPIC_NORMALIZATION = {
    # Physics - Thermodynamics variants
    "Thermodynamics - Joule-Thomson Effect": "Joule-Thomson Effect",
    "Thermodynamics - Enthalpy": "Enthalpy",
    "Cyclic Processes": "Thermodynamic Processes",
    "Adiabatic Process": "Thermodynamic Processes",
    "Entropy Change": "Entropy",
    "Degrees of Freedom": "Kinetic Theory",
    
    # Physics - Optics variants
    "Double Slit Interference": "Interference",
    "Young's Double Slit": "Interference",
    "Lens Formula": "Geometrical Optics",
    "Polarization": "Wave Optics",
    
    # Physics - Electromagnetism variants
    "Magnetic Flux": "Electromagnetism",
    "Magnetic Field due to Current": "Electromagnetism",
    "Electric Field": "Electrostatics",
    "LC Circuit": "Electromagnetic Induction",
    "Magnetic Moment": "Magnetism",
    
    # Physics - Mechanics variants
    "Potential Energy": "Energy Conservation",
    "Projectile Motion": "Kinematics",
    "Simple Harmonic Motion": "Oscillations",
    
    # Physics - Modern Physics variants
    "Uncertainty Principle": "Quantum Mechanics",
    "de Broglie Wavelength": "Wave-Particle Duality",
    "Blackbody Radiation": "Thermal Radiation",
    "Photoelectric Effect": "Quantum Mechanics",
    "Photon Absorption": "Atomic Spectra",
    "Bohr Model": "Atomic Structure",
    
    # Physics - Nuclear variants
    "Radioactive Decay": "Nuclear Physics",
    "Half-Life": "Nuclear Physics",
    "Nuclear Density": "Nuclear Physics",
    "Nuclear Fission Energy": "Nuclear Physics",
    
    # Physics - Waves variants
    "Wave Equation": "Wave Motion",
    "Doppler Effect": "Wave Phenomena",
    "Rayleigh Scattering": "Scattering",
    "Thomson Scattering": "Scattering",
    
    # Chemistry - Organic variants
    "Reaction Mechanisms": "Organic Reactions",
    "Reaction Sequences": "Organic Reactions",
    "Nucleophilic Substitution": "Organic Reactions",
    "IUPAC Nomenclature": "Organic Nomenclature",
    "Isomerism": "Stereochemistry",
    "Aromaticity": "Aromatic Chemistry",
    "Basicity of Amines": "Amines",
    "Synthesis": "Organic Synthesis",
    
    # Chemistry - Physical variants
    "Gibbs Free Energy": "Thermodynamics",
    "Vapour Pressure": "Solutions",
    "Colligative Properties": "Solutions",
    "Le Chatelier's Principle": "Chemical Equilibrium",
    "Activation Energy": "Chemical Kinetics",
    "pH Calculation": "Acids and Bases",
    "Solubility": "Ionic Equilibrium",
    "Solubility Product": "Ionic Equilibrium",
    
    # Chemistry - Inorganic variants
    "Crystal Lattice": "Solid State",
    "Bond Order": "Chemical Bonding",
    "Molecular Orbital Theory": "Chemical Bonding",
    "Boron Compounds": "p-Block Elements",
    
    # Biology - Genetics variants
    "Pedigree Analysis": "Genetics",
    "Hardy-Weinberg Principle": "Population Genetics",
    "Mendelian Inheritance": "Genetics",
    "X-linked Inheritance": "Genetics",
    "Gene Interaction": "Genetics",
    "Genetic Code": "Molecular Genetics",
    "Transcription": "Gene Expression",
    "Lac Operon": "Gene Regulation",
    
    # Biology - Evolution variants
    "Natural Selection": "Evolution",
    "Sexual Selection": "Evolution",
    "Population Genetics": "Evolution",
    
    # Biology - Cell Biology variants
    "Meiosis": "Cell Division",
    "Cell Cycle and Cell Division": "Cell Division",
    
    # Biology - Ecology variants
    "Population Pyramids": "Population Ecology",
    "Carbon Trading": "Environmental Science",
    
    # Biology - Physiology variants
    "Action Potential": "Neural Physiology",
    "Immune System": "Immunology",
    "Seed Germination": "Plant Development",
    "Bacterial Growth": "Microbiology",
    "Enzyme Kinetics": "Enzymology",
    
    # Mathematics variants
    "Conditional Probability": "Probability",
    "Definite Integrals": "Integration",
    "Continuity and Differentiability": "Continuity",
    "Differentiability": "Continuity",
    "Limits": "Limits and Continuity",
    "Continuity": "Limits and Continuity",
    "Matrix Equations": "Linear Algebra",
    "Determinants": "Linear Algebra",
    "System of Equations": "Linear Algebra",
    "Equivalence Relations": "Set Theory",
    "Set Operations": "Set Theory",
    "Diophantine Equations": "Number Theory",
    "Roots of Equations": "Algebra",
    "Quadratic Equations": "Algebra",
    "Functional Equations": "Functions",
    "Vector Triple Product": "Vectors",
    "Scalar Triple Product": "Vectors",
    
    # General/Reasoning variants
    "Cryptography": "Logical Reasoning",
    "Milgram Experiment": "General Knowledge",
    "Spatial Reasoning": "Spatial Aptitude",
    "Data Interpretation": "Quantitative Reasoning",
}

def normalize_topic(topic):
    """Normalize topic name using the mapping"""
    if not topic:
        return topic
    # Strip whitespace and check mapping
    topic = topic.strip()
    return TOPIC_NORMALIZATION.get(topic, topic)

# ==========================================
# CONFIGURATION
# ==========================================

# NEST exam typically has ~75-80 questions per shift
# Based on actual data analysis: avg ~75 questions per paper
TOTAL_NEST_QUESTIONS = 80

# NEST subject distribution (based on analyzing all_merged_cleaned.json actual data)
# Subjects: Mathematics, Physics, Biology, Chemistry, General
# Approximate distribution per paper based on historical averages
SECTION_DISTRIBUTION = {
    "Mathematics": 20,      # ~25% of questions
    "Physics": 20,          # ~25% of questions
    "Biology": 19,          # ~24% of questions
    "Chemistry": 18,        # ~23% of questions
    "General": 3,           # ~3% of questions (logical reasoning, etc.)
}
# Sum verification: 20+20+19+18+3 = 80 ✓

ERA_WEIGHTS = {
    "ancient": 0.05,      # Pre-2011
    "early": 0.3,         # 2011-2014
    "stable": 0.65,       # 2015-2019
    "recent": 0.85,       # 2020-2022
    "current": 1.0        # 2023+
}

# NEST syllabus hasn't had major structural changes like GATE,
# but we track any known shifts
SYLLABUS_CHANGES = {
    2020: ["COVID year - format adjustments"],
    2021: ["No exam held"],  # Gap year
}

# Invalid subjects - topics that shouldn't be in NEST
INVALID_SUBJECTS = set()  # NEST covers all sciences; no subjects to exclude

# Short abbreviations - exact match only to avoid false positives
INVALID_SUBJECT_ABBREVS = set()

INVALID_TOPICS = set()  # NEST is broad; no topics to explicitly exclude

# Official NEST Syllabus - Core Topics by Subject
CORE_TOPICS = {
    "Physics": [
        "Mechanics", "Kinematics", "Dynamics", "Energy Conservation",
        "Gravitation", "Oscillations", "Waves", "Wave Motion",
        "Thermodynamics", "Kinetic Theory", "Thermodynamic Processes", "Entropy",
        "Electrostatics", "Electric Field", "Gauss Law", "Capacitance",
        "Current Electricity", "Ohm's Law", "Kirchhoff's Laws",
        "Magnetism", "Electromagnetism", "Electromagnetic Induction",
        "Optics", "Geometrical Optics", "Wave Optics", "Interference", "Diffraction",
        "Modern Physics", "Quantum Mechanics", "Atomic Structure",
        "Nuclear Physics", "Radioactive Decay",
        "Semiconductor Physics", "Semiconductors",
        "Fluid Mechanics", "Electromagnetic Waves",
        "Units and Dimensions", "Dimensional Analysis",
        "Astrophysics", "Astronomy and Cosmology", "Black Holes",
    ],
    "Chemistry": [
        "Organic Chemistry", "Organic Reactions", "Stereochemistry",
        "Aromatic Chemistry", "Organic Synthesis", "Amines",
        "Inorganic Chemistry", "p-Block Elements", "Coordination Chemistry",
        "Chemical Bonding", "Molecular Orbital Theory",
        "Physical Chemistry", "Chemical Kinetics", "Chemical Equilibrium",
        "Thermodynamics", "Electrochemistry", "Solutions",
        "Acids and Bases", "Ionic Equilibrium",
        "Atomic Structure", "Periodic Table",
        "Solid State", "Surface Chemistry",
        "General Chemistry", "Biomolecules", "Carbohydrates",
        "Metallurgy", "Analytical Chemistry",
        "Nuclear Chemistry",
    ],
    "Biology": [
        "Genetics", "Molecular Genetics", "Gene Expression", "Gene Regulation",
        "Evolutionary Biology", "Evolution", "Natural Selection",
        "Molecular Biology", "DNA as Genetic Material",
        "Cell Biology", "Cell Division",
        "Ecology", "Population Ecology", "Ecosystem",
        "Biochemistry", "Enzymology",
        "Plant Physiology", "Plant Development", "Photosynthesis",
        "Human Physiology", "Neural Physiology",
        "Immunology", "Microbiology",
        "Biotechnology", "Diversity of Living Organisms",
        "Animal Physiology", "Biomolecules",
    ],
    "Mathematics": [
        "Calculus", "Integration", "Limits and Continuity", "Differential Equations",
        "Algebra", "Polynomials", "Sequences and Series",
        "Coordinate Geometry", "Parabola", "Locus",
        "Trigonometry", "Complex Numbers",
        "Number Theory", "Combinatorics",
        "Probability", "Statistics",
        "Linear Algebra", "Matrices", "Vectors",
        "Geometry", "Mensuration",
        "Set Theory", "Functions",
        "Binomial Theorem",
        "Quadratic Equations",
    ],
    "General": [
        "Logical Reasoning", "Spatial Aptitude",
        "General Knowledge", "Quantitative Reasoning",
        "Data Interpretation", "Arithmetic",
        "Computer Science",
    ]
}

# Topic to subject mapping for edge cases
TOPIC_TO_SUBJECT_MAP = {
    # Physics topics
    "Mechanics": "Physics", "Kinematics": "Physics", "Dynamics": "Physics",
    "Thermodynamics": "Physics", "Optics": "Physics", "Electrostatics": "Physics",
    "Electromagnetism": "Physics", "Quantum Mechanics": "Physics",
    "Nuclear Physics": "Physics", "Waves": "Physics", "Oscillations": "Physics",
    "Fluid Mechanics": "Physics", "Gravitation": "Physics",
    "Semiconductors": "Physics", "Magnetism": "Physics",
    "Astrophysics": "Physics", "Units and Dimensions": "Physics",
    "Dimensional Analysis": "Physics",
    
    # Chemistry topics
    "Organic Chemistry": "Chemistry", "Inorganic Chemistry": "Chemistry",
    "Chemical Bonding": "Chemistry", "Chemical Kinetics": "Chemistry",
    "Chemical Equilibrium": "Chemistry", "Electrochemistry": "Chemistry",
    "Solid State": "Chemistry", "Solutions": "Chemistry",
    "Coordination Chemistry": "Chemistry", "Metallurgy": "Chemistry",
    "Acids and Bases": "Chemistry", "p-Block Elements": "Chemistry",
    
    # Biology topics
    "Genetics": "Biology", "Evolution": "Biology", "Ecology": "Biology",
    "Cell Biology": "Biology", "Molecular Biology": "Biology",
    "Biochemistry": "Biology", "Immunology": "Biology",
    "Microbiology": "Biology", "Biotechnology": "Biology",
    "Plant Physiology": "Biology", "Human Physiology": "Biology",
    "Animal Physiology": "Biology",
    
    # Mathematics topics
    "Calculus": "Mathematics", "Algebra": "Mathematics",
    "Probability": "Mathematics", "Statistics": "Mathematics",
    "Trigonometry": "Mathematics", "Geometry": "Mathematics",
    "Number Theory": "Mathematics", "Combinatorics": "Mathematics",
    "Linear Algebra": "Mathematics", "Vectors": "Mathematics",
    "Complex Numbers": "Mathematics", "Set Theory": "Mathematics",
    "Coordinate Geometry": "Mathematics", "Differential Equations": "Mathematics",
    
    # General topics
    "Logical Reasoning": "General", "Data Interpretation": "General",
    "Spatial Reasoning": "General", "Computer Science": "General",
    "Arithmetic": "General",
}

TOPIC_GRANULARITY_MAP = {
    "Organic Reactions": ["Nucleophilic Substitution", "Electrophilic Addition", 
                          "Elimination", "Rearrangement", "Free Radical"],
    "Genetics": ["Pedigree Analysis", "Mendelian Inheritance", "X-linked Inheritance",
                 "Gene Interaction", "Epistasis"],
    "Thermodynamics": ["Cyclic Processes", "Adiabatic Process", "Isothermal Process",
                       "Carnot Cycle", "Entropy Change"],
    "Optics": ["Interference", "Diffraction", "Polarization", "Geometrical Optics",
               "Lens Formula", "Mirror Formula"],
    "Calculus": ["Limits", "Continuity", "Differentiability", "Integration",
                 "Definite Integrals", "Differential Equations"],
    "Nuclear Physics": ["Radioactive Decay", "Half-Life", "Nuclear Fission",
                        "Nuclear Fusion", "Nuclear Density"],
    "Chemical Kinetics": ["Rate Laws", "Activation Energy", "Reaction Order",
                          "Arrhenius Equation"],
    "Evolution": ["Natural Selection", "Sexual Selection", "Hardy-Weinberg Principle",
                  "Speciation", "Genetic Drift"],
}

MIN_APPEARANCES = 2
MIN_LAST_SEEN_YEAR = 2014
MIN_CONFIDENCE_THRESHOLD = 0.15

# Multi-paper years mapping (shifts in NEST)
MULTI_PAPER_YEARS = {
    2020: 2,   # 2 shifts
    2022: 2,   # 2 shifts
    2023: 2,   # 2 shifts
    2024: 2,   # 2 shifts
}

# Diagnostic mode flag
DIAGNOSTIC_MODE = True


def get_era_weight(year):
    if year < 2011:
        return ERA_WEIGHTS["ancient"]
    elif 2011 <= year <= 2014:
        return ERA_WEIGHTS["early"]
    elif 2015 <= year <= 2019:
        return ERA_WEIGHTS["stable"]
    elif 2020 <= year <= 2022:
        return ERA_WEIGHTS["recent"]
    else:
        return ERA_WEIGHTS["current"]

def get_syllabus_penalty(topic, year):
    for invalid in INVALID_TOPICS:
        if invalid.lower() in topic.lower():
            if year >= 2015:
                return 0.0
            return 0.1
    return 1.0

def normalize_subject(subject_raw, topic_raw=None):
    """
    Normalize subject names to match official NEST syllabus categories.
    Also uses topic information to help classify when subject is ambiguous.
    """
    if not subject_raw:
        # Try to infer from topic if subject is missing
        if topic_raw and topic_raw in TOPIC_TO_SUBJECT_MAP:
            return TOPIC_TO_SUBJECT_MAP[topic_raw]
        return None
    
    subject = str(subject_raw).strip()
    
    # NEST has no invalid subjects - all 5 subjects are valid
    # (Physics, Chemistry, Biology, Mathematics, General)
    
    # Mapping covering all variations found in NEST papers
    # NEST has 5 subjects: Mathematics, Physics, Biology, Chemistry, General
    mapping = {
        # Mathematics
        "mathematics": "Mathematics",
        "maths": "Mathematics",
        "math": "Mathematics",
        
        # Physics
        "physics": "Physics",
        "phy": "Physics",
        
        # Biology
        "biology": "Biology",
        "bio": "Biology",
        
        # Chemistry
        "chemistry": "Chemistry",
        "chem": "Chemistry",
        
        # General
        "general": "General",
        "general knowledge": "General",
        "logical reasoning": "General",
        "reasoning": "General",
        "aptitude": "General",
        "computer science": "General",
    }
    
    # Try exact match first (case-insensitive)
    subject_lower = subject.lower().strip()
    if subject_lower in mapping:
        return mapping[subject_lower]
    
    # Try substring match (case-insensitive) - be more careful here
    for key, val in mapping.items():
        if len(key) >= 3 and key in subject_lower:
            return val
    
    # Direct match with SECTION_DISTRIBUTION
    if subject in SECTION_DISTRIBUTION:
        return subject
    
    # Try to infer from topic if available
    if topic_raw:
        topic_str = str(topic_raw).strip()
        if topic_str in TOPIC_TO_SUBJECT_MAP:
            return TOPIC_TO_SUBJECT_MAP[topic_str]
        # Check topic substrings
        for topic_key, subj in TOPIC_TO_SUBJECT_MAP.items():
            if topic_key.lower() in topic_str.lower():
                return subj
    
    return None

def split_combined_subject(subject_raw, topic, marks, question_text="", verbose=False):
    """
    Handle combined subjects (not applicable for NEST).
    NEST subjects are always distinct (Physics, Chemistry, Biology, Mathematics, General).
    Returns None to indicate no splitting needed.
    """
    return None  # NEST doesn't have combined subjects

def _original_split_combined_subject(subject_raw, topic, marks, question_text="", verbose=False):
    """
    Original GATE logic preserved for reference.
    Handle combined subjects like 'Programming and Data Structures'.
    Returns list of (subject, topic, marks) tuples.
    Uses expanded keyword matching on both topic and question text.
    """
    subject = str(subject_raw).strip() if subject_raw else ""
    
    # Check if this is a combined subject
    for combined, split_ratio in COMBINED_SUBJECT_SPLIT.items():
        if combined.lower() in subject.lower():
            # Combine topic and question text for better classification
            topic_lower = topic.lower() if topic else ""
            q_text_lower = question_text.lower() if question_text else ""
            combined_text = f"{topic_lower} {q_text_lower}"
            
            # Expanded keywords for Programming (C language specific)
            programming_keywords = [
                'pointer', 'pointers', 'recursion', 'recursive', 'function', 'functions',
                'array', 'arrays', 'string', 'strings', 'scope', 'parameter', 'parameters',
                'c programming', 'memory', 'struct', 'structure', 'scanf', 'printf',
                'variable', 'variables', 'loop', 'for loop', 'while loop', 'condition',
                'operator', 'operators', 'bitwise', 'preprocessor', 'macro', 'typedef',
                'storage class', 'static', 'extern', 'auto', 'register', 'volatile',
                'malloc', 'calloc', 'realloc', 'free', 'dynamic allocation',
                'pass by value', 'pass by reference', 'call by', 'return value',
                'main()', 'int main', 'void main', 'argc', 'argv', 'command line',
                'header file', '#include', '#define', 'switch', 'case', 'break',
                'continue', 'goto', 'sizeof', 'type casting', 'type conversion',
                'union', 'enum', 'enumeration', 'file handling', 'fopen', 'fclose',
                'fprintf', 'fscanf', 'fread', 'fwrite', 'getchar', 'putchar'
            ]
            
            # Expanded keywords for Data Structures
            ds_keywords = [
                'tree', 'trees', 'graph', 'graphs', 'stack', 'stacks', 'queue', 'queues',
                'linked list', 'linked lists', 'singly linked', 'doubly linked', 'circular list',
                'heap', 'heaps', 'min heap', 'max heap', 'binary heap', 'priority queue',
                'hash', 'hashing', 'hash table', 'hash function', 'collision',
                'bst', 'binary search tree', 'avl', 'avl tree', 'red-black', 'red black',
                'binary tree', 'binary trees', 'traversal', 'inorder', 'preorder', 'postorder',
                'level order', 'bfs', 'dfs', 'breadth first', 'depth first',
                'sorting', 'sort', 'merge sort', 'quick sort', 'heap sort', 'bubble sort',
                'insertion sort', 'selection sort', 'radix sort', 'counting sort',
                'searching', 'search', 'linear search', 'binary search',
                'b-tree', 'b tree', 'b+ tree', 'trie', 'suffix tree', 'segment tree',
                'adjacency matrix', 'adjacency list', 'graph representation',
                'spanning tree', 'mst', 'minimum spanning', 'dijkstra', 'bellman',
                'shortest path', 'topological', 'strongly connected', 'cycle detection',
                'array implementation', 'linked implementation', 'node', 'nodes',
                'insert', 'delete', 'push', 'pop', 'enqueue', 'dequeue',
                'height', 'depth', 'level', 'leaf', 'root', 'parent', 'child', 'sibling'
            ]
            
            # Count keyword matches
            prog_matches = sum(1 for kw in programming_keywords if kw in combined_text)
            ds_matches = sum(1 for kw in ds_keywords if kw in combined_text)
            
            if DIAGNOSTIC_MODE and verbose:
                print(f"      🔍 Split analysis: topic='{topic[:50]}...' prog={prog_matches}, ds={ds_matches}")
            
            # Use match counts for more nuanced decision
            if prog_matches > ds_matches and prog_matches >= 1:
                return [("Programming", topic, marks)]
            elif ds_matches > prog_matches and ds_matches >= 1:
                return [("Data Structures", topic, marks)]
            elif prog_matches == ds_matches and prog_matches > 0:
                # Equal matches - use ratio split
                return [
                    ("Programming", topic, marks * split_ratio["Programming"]),
                    ("Data Structures", topic, marks * split_ratio["Data Structures"])
                ]
            else:
                # No keyword matches - use ratio split
                return [
                    ("Programming", topic, marks * split_ratio["Programming"]),
                    ("Data Structures", topic, marks * split_ratio["Data Structures"])
                ]
    
    return None  # Not a combined subject

def analyze_data_quality(raw_data, processed_data):
    """Analyze and report data quality issues."""
    print("\n📊 DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Check for expected paper counts per year
    year_paper_counts = defaultdict(lambda: defaultdict(int))
    for q in raw_data:
        year = q.get('year')
        paper_id = q.get('shift', 'Shift 1')
        if year:
            year_paper_counts[year][paper_id] += 1
    
    print("\n📋 Paper completeness check (expected ~65 questions per paper):")
    issues = []
    for year in sorted(year_paper_counts.keys()):
        for paper_id, count in year_paper_counts[year].items():
            status = "✅" if 60 <= count <= 70 else "⚠️"
            if count < 60:
                issues.append((year, paper_id, count))
            print(f"   {status} {year} {paper_id}: {count} questions")
    
    if issues:
        print(f"\n⚠️  {len(issues)} papers with potential data issues!")
    
    # Subject coverage check
    print("\n📋 Subject coverage in processed data:")
    year_subject_counts = defaultdict(lambda: defaultdict(int))
    for q in processed_data:
        year_subject_counts[q['year']][q['subject']] += 1
    
    for year in sorted(year_subject_counts.keys())[-5:]:  # Last 5 years
        print(f"\n   {year}:")
        for subject in SECTION_DISTRIBUTION:
            count = year_subject_counts[year].get(subject, 0)
            expected = SECTION_DISTRIBUTION[subject]
            # For multiple papers in a year, multiply expected
            num_papers = len([p for p in year_paper_counts.get(year, {}).keys()])
            expected_total = expected * max(1, num_papers // 65 + 1) if num_papers > 65 else expected
            status = "✅" if count > 0 else "❌"
            print(f"      {status} {subject:<25s}: {count:3d}")

def load_and_process_data(file_path="all_merged_cleaned.json", verbose=True):
    """Load and process GATE question data with comprehensive validation."""
    print("\n📂 Loading and processing data...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    raw_data = []
    
    try:
        parsed = json.loads(content)
        raw_data = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    raw_data.extend(obj if isinstance(obj, list) else [obj])
                except:
                    continue
        
        if not raw_data:
            depth = 0
            obj_start = 0
            for i, ch in enumerate(content):
                if ch == '{':
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(content[obj_start:i+1])
                            raw_data.extend(obj if isinstance(obj, list) else [obj])
                        except:
                            pass
    
    print(f"   ✓ Loaded {len(raw_data)} raw questions")
    
    processed = []
    stats = {
        'excluded_da': 0,
        'excluded_invalid_subject': 0,
        'excluded_invalid_topic': 0,
        'excluded_old_format': 0,
        'excluded_insufficient_text': 0,
        'split_combined': 0,
        'split_to_programming': 0,
        'split_to_ds': 0,
        'split_ratio': 0  # Count of ratio-based splits
    }
    
    # Track unmapped subjects for debugging
    unmapped_subjects = defaultdict(int)
    unmapped_examples = defaultdict(list)
    
    for q in raw_data:
        year = q.get('year')
        if not year:
            year = 2024  # Default year if missing
        year = int(year)
        
        # NEST uses 'shift' not file_name
        paper_id = q.get('shift', '').upper()
        
        # DA filter removed - now includes all 2024-2025 CSE data
        # (No DA papers in our dataset)
        
        # Get raw values
        raw_subject = q.get('subject')
        topic = q.get('chapter') or ""
        micro_topic_raw = q.get('topic', '') or topic  # fallback to topic
        q_text = q.get('question_text') or ""
        
        # IMPORTANT: Check for combined subjects FIRST (before normalize_subject)
        # This ensures "Programming and Data Structures" gets properly split
        if raw_subject:
            raw_subject_lower = str(raw_subject).lower()
            if 'programming' in raw_subject_lower and 'data structure' in raw_subject_lower:
                split_result = split_combined_subject(raw_subject, topic, q.get('marks', 1), q_text)
                if split_result:
                    # Handle split subjects and track where each goes
                    for split_subj, split_topic, split_marks in split_result:
                        if split_subj in SECTION_DISTRIBUTION:
                            stats['split_combined'] += 1
                            # Track split destination
                            if split_subj == "Programming":
                                stats['split_to_programming'] += 1
                            elif split_subj == "Data Structures":
                                stats['split_to_ds'] += 1
                            # Track if it was a ratio split (both subjects in result)
                            if len(split_result) > 1:
                                stats['split_ratio'] += 1
                            processed.append({
                                'year': year,
                                'subject': split_subj,
                                'topic': split_topic or split_subj,
                                'micro_topic': normalize_topic(micro_topic_raw) if micro_topic_raw else (split_topic or split_subj),
                                'question_text': q_text,
                                'question_type': q.get('question_type', 'MCQ'),
                                'choices': q.get('choices') or [],
                                'answer': q.get('answer') or '',
                                'marks': split_marks,
                                'paper_id': q.get('shift', ''),
                                'era_weight': get_era_weight(year),
                                'syllabus_penalty': get_syllabus_penalty(split_topic or "", year),
                                'combined_weight': get_era_weight(year) * get_syllabus_penalty(split_topic or "", year),
                                'from_combined': True  # Track origin
                            })
                    continue
        
        # Now try to normalize subject (for non-combined subjects)
        subject = normalize_subject(raw_subject, topic)
        
        # If subject still not found, try inferring from topic
        if subject is None and topic:
            subject = TOPIC_TO_SUBJECT_MAP.get(topic)
            if subject is None:
                # Try partial matching
                for topic_key, subj in TOPIC_TO_SUBJECT_MAP.items():
                    if topic_key.lower() in topic.lower():
                        subject = subj
                        break
        
        if subject is None or subject not in SECTION_DISTRIBUTION:
            stats['excluded_invalid_subject'] += 1
            if raw_subject:
                unmapped_subjects[str(raw_subject)] += 1
                if len(unmapped_examples[str(raw_subject)]) < 2:
                    unmapped_examples[str(raw_subject)].append({
                        'topic': topic,
                        'year': year,
                        'paper': q.get('shift', 'Shift 1')
                    })
            continue
        
        # Use topic from data, or fall back to subject
        if not topic:
            topic = subject
        
        # Apply topic normalization to merge duplicate variants
        topic = normalize_topic(topic)
        
        # Check for invalid topics
        is_invalid_topic = False
        for invalid in INVALID_TOPICS:
            if invalid.lower() in topic.lower():
                is_invalid_topic = True
                break
        
        if is_invalid_topic:
            stats['excluded_invalid_topic'] += 1
            continue
        
        q_text = q.get('question_text') or ""
        
        q_type = 'MCQ'  # NEST is primarily MCQ format
        choices = q.get('choices') or {}
        if isinstance(choices, dict) and len(choices) > 0:
            q_type = "MCQ"
        
        era_weight = get_era_weight(year)
        syllabus_penalty = get_syllabus_penalty(topic, year)
        combined_weight = era_weight * syllabus_penalty
        
        # Normalize micro_topic
        micro_topic = normalize_topic(micro_topic_raw) if micro_topic_raw else topic
        
        processed.append({
            'year': year,
            'subject': subject,
            'topic': topic,
            'micro_topic': micro_topic,
            'question_text': q_text,
            'question_type': q_type,
            'choices': q.get('choices') or [],
            'answer': q.get('answer') or '',
            'marks': q.get('marks', 1),
            'paper_id': q.get('shift', ''),
            'era_weight': era_weight,
            'syllabus_penalty': syllabus_penalty,
            'combined_weight': combined_weight,
            'difficulty': q.get('difficulty'),
            'correct_answer': q.get('correct_answer')
        })
    
    print(f"   ✓ Processed {len(processed)} valid questions")
    print(f"   ✓ Excluded: DA={stats['excluded_da']}, "
          f"Invalid subjects={stats['excluded_invalid_subject']}, "
          f"Invalid topics={stats['excluded_invalid_topic']}")
    if stats['split_combined'] > 0:
        print(f"   ✓ Split {stats['split_combined']} combined subject entries:")
        if DIAGNOSTIC_MODE:
            print(f"      → Programming: {stats['split_to_programming']} entries")
            print(f"      → Data Structures: {stats['split_to_ds']} entries")
            print(f"      → Ratio-based splits: {stats['split_ratio']//2} questions")
    
    # Show unmapped subjects if any (helps debug data issues)
    if unmapped_subjects and verbose:
        print(f"\n   ⚠️  Top unmapped subjects (need to add to mapping):")
        for subj, count in sorted(unmapped_subjects.items(), key=lambda x: -x[1])[:15]:
            examples = unmapped_examples.get(subj, [])
            example_str = ""
            if examples:
                ex = examples[0]
                example_str = f" (e.g., topic='{ex['topic']}', year={ex['year']})"
            print(f"      {count:3d}× '{subj}'{example_str}")
    
    # Year distribution
    year_dist = Counter(q['year'] for q in processed)
    print(f"\n   📅 Year distribution:")
    for year in sorted(year_dist.keys()):
        print(f"      {year}: {year_dist[year]} questions")
    
    # Subject distribution check
    subject_dist = Counter(q['subject'] for q in processed)
    print(f"\n   📊 Subject distribution (all years combined):")
    total_processed = len(processed)
    for subject in SECTION_DISTRIBUTION:
        count = subject_dist.get(subject, 0)
        pct = (count / total_processed * 100) if total_processed > 0 else 0
        expected_pct = (SECTION_DISTRIBUTION.get(subject, 0) / TOTAL_NEST_QUESTIONS * 100)
        status = "✅" if abs(pct - expected_pct) <= 5 else ("⚠️" if count > 0 else "❌")
        print(f"      {status} {subject:<25s}: {count:4d} ({pct:5.1f}%, expected ~{expected_pct:.1f}%)")
    
    # Analyze data quality
    if verbose:
        analyze_data_quality(raw_data, processed)
    
    return processed

# ==========================================
# ADVANCED FEATURE ENGINEERING
# ==========================================

def extract_elite_features(topic_data, all_topics_data, target_year=2025):
    years = sorted(topic_data['years'])
    counts = [topic_data['year_counts'][y] for y in years]
    weights = [get_era_weight(y) for y in years]
    
    if not counts:
        return np.zeros(55)
    
    features = []
    
    # TEMPORAL FEATURES (15)
    w_mean = np.average(counts, weights=weights) if weights else np.mean(counts)
    recent_mean = np.mean(counts[-3:]) if len(counts) >= 3 else w_mean
    std_dev = np.std(counts) if len(counts) > 1 else 0.0
    features.extend([w_mean, recent_mean, std_dev])
    
    if len(counts) >= 3:
        x = np.arange(len(counts))
        coeffs = np.polyfit(x, counts, 1, w=np.array(weights))
        slope = coeffs[0]
        
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((np.array(counts) - y_pred) ** 2)
        ss_tot = np.sum((np.array(counts) - np.mean(counts)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        momentum = counts[-1] - counts[-2] if len(counts) >= 2 else 0
    else:
        slope, r2, momentum = 0, 0, 0
    
    features.extend([slope, r2, momentum])
    
    for alpha in [0.3, 0.5, 0.7, 0.85, 0.95]:
        if len(counts) > 0:
            ema_weights = np.array([alpha ** i for i in range(len(counts))][::-1])
            ema = np.sum(np.array(counts) * ema_weights) / ema_weights.sum()
        else:
            ema = 0
        features.append(ema)
    
    if len(counts) >= 3:
        diffs = np.diff(counts)
        acceleration = np.mean(np.diff(diffs)) if len(diffs) > 1 else 0
        accel_std = np.std(np.diff(diffs)) if len(diffs) > 1 else 0
    else:
        acceleration, accel_std = 0, 0
    features.extend([acceleration, accel_std])
    
    last_seen = max(years) if years else 2000
    years_since = target_year - last_seen
    recency_score = np.exp(-years_since / 3.0)
    features.extend([years_since, recency_score])
    
    # STATISTICAL FEATURES (10)
    features.extend([
        min(counts),
        max(counts),
        max(counts) - min(counts),
        np.percentile(counts, 25) if len(counts) >= 4 else min(counts),
        np.percentile(counts, 75) if len(counts) >= 4 else max(counts)
    ])
    
    cv = std_dev / w_mean if w_mean > 0 else 0
    iqr = np.percentile(counts, 75) - np.percentile(counts, 25) if len(counts) >= 4 else std_dev
    variance = np.var(counts)
    
    if SCIPY_AVAILABLE and len(counts) >= 4:
        skewness = stats.skew(counts)
        kurtosis = stats.kurtosis(counts)
    else:
        skewness, kurtosis = 0, 0
    
    features.extend([cv, iqr, variance, skewness, kurtosis])
    
    # ERA-BASED FEATURES (5)
    era_avgs = []
    for era_name, era_range in [
        ('early', (2011, 2014)),
        ('stable', (2015, 2019)),
        ('recent', (2020, 2022)),
        ('current', (2023, 2025))
    ]:
        era_vals = [c for y, c in zip(years, counts) 
                    if era_range[0] <= y <= era_range[1]]
        era_avg = np.mean(era_vals) if era_vals else 0
        era_avgs.append(era_avg)
    
    era_transition = era_avgs[-1] - era_avgs[-2] if len(era_avgs) >= 2 else 0
    features.extend(era_avgs + [era_transition])
    
    # CONSISTENCY FEATURES (5)
    total_years = max(years) - min(years) + 1 if len(years) > 1 else 1
    appearance_rate = len(years) / total_years
    
    consecutive = 0
    max_consecutive = 0
    for i in range(len(years) - 1):
        if years[i+1] == years[i] + 1:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    
    stability = 1 / (1 + cv)
    recent_stability = 1 / (1 + np.std(counts[-3:]) / np.mean(counts[-3:])) if len(counts) >= 3 and np.mean(counts[-3:]) > 0 else 0.5
    
    features.extend([appearance_rate, max_consecutive, stability, recent_stability, len(years)])
    
    # SYLLABUS-AWARE FEATURES (8)
    subject = topic_data.get('subject', '')
    topic = topic_data.get('topic', '')
    
    is_core = 1.0 if any(topic in CORE_TOPICS.get(s, []) for s in CORE_TOPICS) else 0.0
    
    specificity = 0.5
    for broad, specifics in TOPIC_GRANULARITY_MAP.items():
        if topic in specifics:
            specificity = 1.0
        elif broad in topic:
            specificity = 0.3
    
    top_subjects = ["Mathematics", "Physics", "Biology", "Chemistry"]
    for subj in top_subjects:
        features.append(1.0 if subject == subj else 0.0)
    
    subject_quota = SECTION_DISTRIBUTION.get(subject, 5) / TOTAL_NEST_QUESTIONS
    
    subject_topics = [t for t in all_topics_data if all_topics_data[t].get('subject') == subject]
    if subject_topics:
        subject_means = [np.average(all_topics_data[t]['counts'], 
                                    weights=[get_era_weight(y) for y in all_topics_data[t]['years']])
                        for t in subject_topics if all_topics_data[t]['counts']]
        if subject_means:
            percentile_in_subject = np.searchsorted(sorted(subject_means), w_mean) / len(subject_means)
        else:
            percentile_in_subject = 0.5
    else:
        percentile_in_subject = 0.5
    
    features.extend([is_core, specificity, subject_quota, percentile_in_subject])
    
    # LAST 5 YEARS (5)
    last_5 = [0] * 5
    for i, c in enumerate(counts[-5:]):
        last_5[i] = c
    features.extend(last_5)
    
    # ADVANCED TEMPORAL (7)
    years_active = len(set(years))
    
    if len(years) > 1:
        gaps = [years[i+1] - years[i] for i in range(len(years)-1)]
        avg_gap = np.mean(gaps)
        std_gap = np.std(gaps)
    else:
        avg_gap, std_gap = 0, 0
    
    if avg_gap > 0:
        predicted_next = last_seen + avg_gap
        time_to_predicted = target_year - predicted_next
    else:
        time_to_predicted = 0
    
    poly_1 = w_mean * slope
    poly_2 = recent_mean * recency_score
    poly_3 = stability * is_core
    poly_4 = appearance_rate * subject_quota
    
    features.extend([years_active, avg_gap, std_gap, time_to_predicted,
                    poly_1, poly_2, poly_3, poly_4])
    
    features = features[:55]
    return np.nan_to_num(np.array(features), nan=0.0, posinf=10.0, neginf=-10.0)

# ==========================================
# CONFIDENCE CALIBRATION (IMPROVED)
# ==========================================

# Blend factor: how much raw confidence to preserve (0.3 = 30% raw, 70% calibrated)
CONFIDENCE_BLEND_FACTOR = 0.3

class ConfidenceCalibrator:
    """
    Improved confidence calibrator using isotonic regression.
    Fixes the issue where all predictions showed 84-85% confidence.
    """
    
    def __init__(self):
        self.isotonic_model = None
        self.raw_predictions = []
        self.actual_accuracies = []
        # Keep bins for metrics only, not for calibration
        self.confidence_bins = np.linspace(0.0, 1.0, 21)  # 20 bins instead of 5
        self.bin_accuracies = defaultdict(list)
        self.is_calibrated = False
        
    def add_prediction(self, predicted_conf, actual_accuracy):
        """Add a prediction-accuracy pair for calibration"""
        self.raw_predictions.append(predicted_conf)
        self.actual_accuracies.append(actual_accuracy)
        
        # Also bin for metrics
        for i in range(len(self.confidence_bins) - 1):
            if self.confidence_bins[i] <= predicted_conf < self.confidence_bins[i+1]:
                self.bin_accuracies[i].append(actual_accuracy)
                break
    
    def calibrate(self):
        """Calibrate using isotonic regression for smooth, monotonic mapping"""
        if len(self.raw_predictions) < 5:
            print("      ⚠️  Insufficient data for calibration, using raw confidence")
            self.is_calibrated = False
            return
        
        if ISOTONIC_AVAILABLE:
            # Use isotonic regression for proper calibration
            self.isotonic_model = IsotonicRegression(
                y_min=0.0, 
                y_max=1.0, 
                out_of_bounds='clip'
            )
            X = np.array(self.raw_predictions).reshape(-1, 1)
            y = np.array(self.actual_accuracies)
            
            try:
                self.isotonic_model.fit(X.ravel(), y)
                self.is_calibrated = True
                print(f"      ✓ Isotonic calibration fitted on {len(X)} samples")
            except Exception as e:
                print(f"      ⚠️  Isotonic calibration failed: {e}")
                self.is_calibrated = False
        else:
            # Fallback to improved binning with more granularity
            self._fallback_calibrate()
    
    def _fallback_calibrate(self):
        """Fallback calibration using finer bins when isotonic not available"""
        for bin_idx in range(len(self.confidence_bins) - 1):
            if bin_idx in self.bin_accuracies and len(self.bin_accuracies[bin_idx]) >= 2:
                self.is_calibrated = True
        
    def adjust_confidence(self, raw_confidence):
        """
        Adjust confidence score using calibration + blending.
        Preserves relative differences between predictions.
        
        Args:
            raw_confidence: Value between 0 and 1
            
        Returns:
            Calibrated confidence between 0 and 1 (NOT percentage)
        """
        if not self.is_calibrated:
            return raw_confidence  # Return as 0-1 range
        
        raw_conf_normalized = raw_confidence if raw_confidence <= 1 else raw_confidence / 100
        
        if self.isotonic_model is not None:
            # Use isotonic regression
            try:
                calibrated = self.isotonic_model.predict([raw_conf_normalized])[0]
            except:
                calibrated = raw_conf_normalized
        else:
            # Fallback binning
            calibrated = self._fallback_adjust(raw_conf_normalized)
        
        # Blend raw and calibrated to preserve relative differences
        # This prevents all predictions from clustering at same value
        blended = (CONFIDENCE_BLEND_FACTOR * raw_conf_normalized + 
                   (1 - CONFIDENCE_BLEND_FACTOR) * calibrated)
        
        # Apply variance preservation: scale to maintain spread
        # Add small amount of original variation back
        variation = (raw_conf_normalized - 0.5) * 0.15
        final = blended + variation
        
        # Clamp to valid range (0-1, NOT percentage)
        # Ceiling lowered from 0.95 to 0.85 to prevent over-confidence
        return max(0.10, min(0.85, final))
    
    def _fallback_adjust(self, raw_confidence):
        """Fallback adjustment using binned averages"""
        for i in range(len(self.confidence_bins) - 1):
            if self.confidence_bins[i] <= raw_confidence < self.confidence_bins[i+1]:
                if i in self.bin_accuracies and self.bin_accuracies[i]:
                    return np.mean(self.bin_accuracies[i])
        return raw_confidence
    
    def get_calibration_metrics(self):
        """Get calibration quality metrics"""
        metrics = {
            'total_samples': len(self.raw_predictions),
            'is_calibrated': self.is_calibrated,
            'calibration_method': 'isotonic' if self.isotonic_model else 'binning',
            'blend_factor': CONFIDENCE_BLEND_FACTOR,
            'bin_counts': {},
            'bin_accuracies': {},
            'calibration_error': self._calculate_calibration_error()
        }
        
        # Summarize bins (show only non-empty bins)
        for i in range(len(self.confidence_bins) - 1):
            bin_label = f"{self.confidence_bins[i]:.2f}-{self.confidence_bins[i+1]:.2f}"
            count = len(self.bin_accuracies.get(i, []))
            if count > 0:
                metrics['bin_counts'][bin_label] = count
                metrics['bin_accuracies'][bin_label] = round(
                    np.mean(self.bin_accuracies[i]), 3
                )
        
        return metrics
    
    def _calculate_calibration_error(self):
        """Calculate Expected Calibration Error (ECE)"""
        total_samples = len(self.raw_predictions)
        if total_samples == 0:
            return 0
        
        ece = 0
        for bin_idx in range(len(self.confidence_bins) - 1):
            if bin_idx not in self.bin_accuracies:
                continue
            
            bin_size = len(self.bin_accuracies[bin_idx])
            if bin_size == 0:
                continue
            
            avg_conf = (self.confidence_bins[bin_idx] + self.confidence_bins[bin_idx+1]) / 2
            avg_acc = np.mean(self.bin_accuracies[bin_idx])
            ece += (bin_size / total_samples) * abs(avg_conf - avg_acc)
        
        return round(ece, 4)

# ==========================================
# ELITE ENSEMBLE PREDICTOR
# ==========================================

class EliteNESTPredictor:
    
    def __init__(self):
        self.scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        self.base_models = {}
        self.stacking_model = None
        self.all_topics_data = {}
        self.all_micro_topics_data = {}
        self.metrics = {}
        self.feature_importance = None
        self.calibrator = ConfidenceCalibrator()
        
    def build_base_models(self):
        models = []
        
        if CATBOOST_AVAILABLE:
            models.append(('catboost', cb.CatBoostRegressor(
                iterations=800,
                learning_rate=0.012,
                depth=7,
                loss_function='Poisson',
                random_seed=42,
                verbose=False,
                bootstrap_type='Bayesian',
                bagging_temperature=0.5,
                l2_leaf_reg=3,
                min_data_in_leaf=5
            )))
        
        if XGBOOST_AVAILABLE:
            models.append(('xgboost', xgb.XGBRegressor(
                n_estimators=500,  # Reduced from 700
                learning_rate=0.015,  # Slightly higher
                max_depth=6,  # Reduced from 8
                min_child_weight=5,  # Increased from 3
                subsample=0.7,  # Reduced from 0.8
                colsample_bytree=0.7,  # Reduced from 0.8
                gamma=0.2,  # Increased from 0.1
                reg_alpha=0.5,  # L1 regularization added
                reg_lambda=2.0,  # L2 regularization added
                objective='count:poisson',
                random_state=42,
                tree_method='hist',
                verbosity=0
            )))
        
        if LIGHTGBM_AVAILABLE:
            models.append(('lightgbm', lgb.LGBMRegressor(
                n_estimators=500,  # Reduced from 700
                learning_rate=0.015,
                num_leaves=31,  # Reduced from 63
                min_child_samples=15,  # Increased from 10
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,  # L1 regularization added
                reg_lambda=2.0,  # L2 regularization added
                objective='poisson',
                random_state=42,
                verbose=-1
            )))
        
        if SKLEARN_AVAILABLE:
            models.append(('extratrees', ExtraTreesRegressor(
                n_estimators=400,  # Reduced from 500
                max_depth=10,  # Reduced from 15
                min_samples_split=8,  # Increased from 4
                min_samples_leaf=5,  # Increased from 2
                max_features='sqrt',
                random_state=42,
                n_jobs=2
            )))
            
            models.append(('rf', RandomForestRegressor(
                n_estimators=350,  # Reduced from 400
                max_depth=10,  # Reduced from 14
                min_samples_split=8,  # Increased from 4
                min_samples_leaf=5,  # Increased from 2
                max_features='sqrt',
                random_state=42,
                n_jobs=2
            )))
            
            models.append(('histgb', HistGradientBoostingRegressor(
                max_iter=400,  # Reduced from 500
                learning_rate=0.02,  # Increased from 0.015
                max_depth=8,  # Reduced from 12
                min_samples_leaf=8,  # Increased from 5
                l2_regularization=2.0,  # Increased from 1.0
                random_state=42
            )))
        
        self.base_models = dict(models)
        print(f"   ✓ Built {len(models)} base models")
        return models
    
    def train(self, processed_data):
        print("\n🔧 Training Elite Ensemble...")
        
        topic_data = {}
        micro_topic_data = {}
        
        for q in processed_data:
            key = f"{q['subject']}::{q['topic']}"
            # Initialize the topic_data entry if it doesn't exist
            if key not in topic_data:
                topic_data[key] = {
                    'years': [],
                    'counts': [],
                    'year_counts': {},
                    'subject': '',
                    'topic': ''
                }
            
            topic_data[key]['years'].append(q['year'])
            
            year_key = q['year']
            try:
                year_key = int(year_key)
            except Exception:
                # leave as-is if it cannot be converted
                pass
            topic_data[key]['year_counts'][year_key] = topic_data[key]['year_counts'].get(year_key, 0) + 1
            topic_data[key]['subject'] = q['subject']
            topic_data[key]['topic'] = q['topic']
            
            # Build micro_topic data alongside topic data
            micro_topic = q.get('micro_topic', q['topic'])
            micro_key = f"{q['subject']}::{q['topic']}::{micro_topic}"
            if micro_key not in micro_topic_data:
                micro_topic_data[micro_key] = {
                    'years': [],
                    'counts': [],
                    'year_counts': {},
                    'subject': '',
                    'topic': '',
                    'micro_topic': ''
                }
            micro_topic_data[micro_key]['years'].append(q['year'])
            micro_topic_data[micro_key]['year_counts'][year_key] = micro_topic_data[micro_key]['year_counts'].get(year_key, 0) + 1
            micro_topic_data[micro_key]['subject'] = q['subject']
            micro_topic_data[micro_key]['topic'] = q['topic']
            micro_topic_data[micro_key]['micro_topic'] = micro_topic
        
        for key in topic_data:
            years = sorted(set(topic_data[key]['years']))
            topic_data[key]['years'] = years
            topic_data[key]['counts'] = [topic_data[key]['year_counts'][y] for y in years]
        
        # Finalize micro_topic_data
        for key in micro_topic_data:
            years = sorted(set(micro_topic_data[key]['years']))
            micro_topic_data[key]['years'] = years
            micro_topic_data[key]['counts'] = [micro_topic_data[key]['year_counts'][y] for y in years]
        
        self.all_topics_data = topic_data
        self.all_micro_topics_data = micro_topic_data
        print(f"   ✓ Built {len(topic_data)} topic groups + {len(micro_topic_data)} micro-topic groups")
        
        X_list, y_list = [], []
        
        for key, data in topic_data.items():
            if len(data['years']) < 4:
                continue
            
            for i in range(3, len(data['years'])):
                sample_data = {
                    'years': data['years'][:i],
                    'counts': data['counts'][:i],
                    'year_counts': {y: c for y, c in zip(data['years'][:i], data['counts'][:i])},
                    'subject': data['subject'],
                    'topic': data['topic']
                }
                
                features = extract_elite_features(sample_data, topic_data, data['years'][i])
                target = data['counts'][i]
                
                X_list.append(features)
                y_list.append(target)
        
        if len(X_list) == 0:
            print("❌ Insufficient training data")
            return {}
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"   ✓ Training samples: {len(X)}, Features: {X.shape[1]}")
        
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        self.build_base_models()
        
        if not self.base_models:
            print("⚠️  No ML libraries, using fallback")
            return self._fallback_train(topic_data)
        
        if SKLEARN_AVAILABLE:
            tscv = TimeSeriesSplit(n_splits=8)
            cv_scores = {}
            
            print("\n   📊 Cross-validation scores:")
            for name, model in self.base_models.items():
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    weights = np.array([get_era_weight(2010 + i // 15) for i in train_idx])
                    
                    try:
                        if name in ['catboost', 'xgboost', 'lightgbm']:
                            model.fit(X_train, y_train, sample_weight=weights)
                        else:
                            model.fit(X_train, y_train)
                        
                        preds = np.clip(model.predict(X_val), 0, 25)
                        mae = mean_absolute_error(y_val, preds)
                        scores.append(mae)
                    except Exception as e:
                        print(f"      ⚠️  {name} failed: {e}")
                        continue
                
                if scores:
                    cv_scores[name] = np.mean(scores)
                    print(f"      {name:12s}: MAE = {cv_scores[name]:.4f}")
        
        if SKLEARN_AVAILABLE and len(self.base_models) >= 2:
            print("\n   🧱 Training stacking meta-learner...")
            estimators = [(name, model) for name, model in list(self.base_models.items())[:5]]
            
            self.stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=BayesianRidge(
                    alpha_1=1e-6,
                    alpha_2=1e-6,
                    lambda_1=1e-6,
                    lambda_2=1e-6
                ),
                cv=5,
                n_jobs=2
            )
            
            weights = np.array([get_era_weight(2010 + i // 15) for i in range(len(X))])
            
            # Use cross_val_predict to get out-of-fold metrics and prevent data leakage
            cv_preds = np.clip(cross_val_predict(self.stacking_model, X, y, cv=5, n_jobs=2), 0, 25)
            
            self.stacking_model.fit(X, y, sample_weight=weights)
            
            self.metrics = {
                "MAE": round(mean_absolute_error(y, cv_preds), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y, cv_preds)), 4),
                "R2": round(r2_score(y, cv_preds), 4),
                "Mean_Prediction": round(np.mean(cv_preds), 4)
            }
            
            print(f"\n   🎯 Final Ensemble Metrics:")
            for k, v in self.metrics.items():
                print(f"      {k}: {v}")
        
        return self.metrics
    
    def predict(self, target_year=2025):
        print(f"\n🔮 Generating predictions for {target_year}...")
        
        predictions = []
        
        for key, data in self.all_topics_data.items():
            if len(data['years']) < MIN_APPEARANCES:
                continue
            
            last_seen = max(data['years'])
            if last_seen < MIN_LAST_SEEN_YEAR:
                continue
            
            features = extract_elite_features(data, self.all_topics_data, target_year)
            
            if self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            if self.stacking_model:
                pred = self.stacking_model.predict(features_scaled)[0]
                
                individual_preds = []
                for name, model in self.base_models.items():
                    try:
                        individual_preds.append(model.predict(features_scaled)[0])
                    except:
                        continue
                
                pred_std = np.std(individual_preds) if len(individual_preds) > 1 else 0
            elif self.base_models:
                preds = []
                for model in self.base_models.values():
                    try:
                        preds.append(model.predict(features_scaled)[0])
                    except:
                        continue
                pred = np.mean(preds) if preds else np.mean(data['counts'])
                pred_std = np.std(preds) if len(preds) > 1 else 0
            else:
                pred = self._fallback_predict_single(data)
                pred_std = 0
            
            pred = max(0, pred)
            
            # Subject-specific bias correction based on validation errors
            # To be calibrated after initial runs with NEST data
            SUBJECT_BIAS_CORRECTION = {
                # Add corrections here after analyzing validation results
            }
            if data['subject'] in SUBJECT_BIAS_CORRECTION:
                pred = pred * SUBJECT_BIAS_CORRECTION[data['subject']]
            
            raw_confidence = self._calculate_elite_confidence(data, pred, pred_std, target_year)
            
            # Apply calibration
            calibrated_confidence = self.calibrator.adjust_confidence(raw_confidence)
            
            trend = self._detect_trend(data)
            
            predictions.append({
                "subject": data['subject'],
                "topic": data['topic'],
                "predicted_count": round(pred, 3),
                "confidence": round(calibrated_confidence * 100, 1),
                "raw_confidence": round(raw_confidence * 100, 1),
                "uncertainty": round(pred_std, 3),
                "trend": trend,
                "historical_avg": round(np.mean(data['counts']), 2),
                "recent_avg": round(np.mean(data['counts'][-3:]), 2) if len(data['counts']) >= 3 else round(np.mean(data['counts']), 2),
                "last_seen": last_seen,
                "appearances": len(data['years']),
                "is_core": "✓" if any(data['topic'] in CORE_TOPICS.get(s, []) for s in CORE_TOPICS) else ""
            })
        
        raw_total = sum(p['predicted_count'] for p in predictions)
        print(f"   ✓ Raw predictions: {len(predictions)} topics, {raw_total:.1f} total")
        
        print(f"\n   📐 Normalizing to NEST structure...")
        normalized = self._normalize_to_nest(predictions)
        
        return normalized
    
    def predict_micro_topics(self, topic_predictions, target_year=2025):
        """Micro-topic predictions using weighted recency (backtested winner).
        
        Backtested across 20 methods, 268 parent-year groups, 6 holdout years (2020-2025).
        Winner: count × exp(-years_since / 3.0) with MAE=0.3069 (-3.5% vs proportional baseline).
        
        The top 4 methods are statistically indistinguishable (all t < 1.96), but τ=3.0
        is preferred over τ=4.0 for better stability across recent years and cleaner
        decay semantics: topics decay to ~5% weight after ~9 years.
        """
        print(f"\n🔬 Generating micro-topic predictions for {target_year}...")
        
        if not hasattr(self, 'all_micro_topics_data') or not self.all_micro_topics_data:
            print("   ⚠️  No micro-topic data available")
            return []
        
        # Build lookup: subject::topic -> predicted_count from topic predictions
        topic_pred_lookup = {}
        for p in topic_predictions:
            key = f"{p['subject']}::{p['topic']}"
            topic_pred_lookup[key] = p['predicted_count']
        
        # Group micro_topics by parent topic
        micro_by_parent = defaultdict(list)
        
        # === WEIGHTED RECENCY SCORING ===
        # τ (tau) = 3.0: decay time constant in years
        # - Topics last seen 3 years ago: weight × 0.37 (e^-1)
        # - Topics last seen 6 years ago: weight × 0.14 (e^-2) 
        # - Topics last seen 9 years ago: weight × 0.05 (e^-3)
        RECENCY_TAU = 3.0
        
        for key, data in self.all_micro_topics_data.items():
            # Relaxed filter: allow single-appearance micro-topics
            # Recency decay naturally handles rare topics (their score will be low)
            if len(data['years']) < 1:
                continue
            
            last_seen = max(data['years'])
            # Topics not seen since before 2016 are extremely unlikely to reappear
            # Even with τ=3.0, a 2015 topic in 2027 has weight × 0.02 — negligible
            if last_seen < 2016:
                continue
            
            parent_key = f"{data['subject']}::{data['topic']}"
            total_count = sum(data['counts'])
            years_since = target_year - last_seen
            
            # === CORE FORMULA: Weighted Recency Score ===
            # score = total_historical_count × exp(-years_since_last_seen / τ)
            # This directly models how exam setters think:
            #   - Favor topics that appear frequently (total_count)
            #   - Favor topics that appeared recently (exp decay)
            recency_score = total_count * np.exp(-years_since / RECENCY_TAU)
            
            # === CONFIDENCE (display-only, not used in scoring) ===
            # Three factors with empirically reasonable weights:
            #   - data_factor: more historical data → more confidence
            #   - recency_factor: recently seen → more confidence  
            #   - consistency: appears in most years of its active span → more confidence
            years_active = len(data['years'])
            total_span = max(1, max(data['years']) - min(data['years']) + 1)
            consistency = years_active / total_span
            
            data_factor = min(1.0, years_active / 8)
            recency_factor = np.exp(-years_since / 2.5)
            conf = 0.35 * data_factor + 0.35 * recency_factor + 0.30 * consistency
            
            # Clamp confidence to [15%, 85%] — these weights are heuristic (not backtested)
            confidence = round(min(0.85, max(0.15, conf)) * 100, 1)
            
            trend = self._detect_trend(data)
            
            micro_by_parent[parent_key].append({
                "subject": data['subject'],
                "topic": data['topic'],
                "micro_topic": data['micro_topic'],
                "raw_score": recency_score,
                "confidence": confidence,
                "trend": trend,
                "historical_avg": round(np.mean(data['counts']), 2),
                "recent_avg": round(np.mean(data['counts'][-3:]), 2) 
                    if len(data['counts']) >= 3 
                    else round(np.mean(data['counts']), 2),
                "last_seen": last_seen,
                "appearances": years_active
            })
        
        # === NORMALIZE within parent's predicted quota ===
        # Each parent topic has a predicted question count from the ML ensemble.
        # Distribute that quota across micro-topics proportionally by recency score.
        micro_predictions = []
        
        for parent_key, micros in micro_by_parent.items():
            parent_pred = topic_pred_lookup.get(parent_key, 0)
            raw_total = sum(m['raw_score'] for m in micros)
            
            for m in micros:
                if raw_total > 0 and parent_pred > 0:
                    # predicted_count = (score / sum_of_all_scores) × parent_topic_quota
                    m['predicted_count'] = round(
                        (m['raw_score'] / raw_total) * parent_pred, 3
                    )
                else:
                    m['predicted_count'] = 0
                # Remove internal scoring field from output
                del m['raw_score']
                micro_predictions.append(m)
        
        # Sort by confidence and predicted count (high priority first)
        micro_predictions.sort(
            key=lambda x: (x['confidence'], x['predicted_count']),
            reverse=True
        )
        
        print(f"   ✓ Generated {len(micro_predictions)} micro-topic predictions "
              f"across {len(micro_by_parent)} parent topics")
        print(f"   ✓ Method: Weighted Recency (τ={RECENCY_TAU}), backtested MAE=0.3069")
        return micro_predictions
    
    def _calculate_elite_confidence(self, data, prediction, uncertainty, target_year):
        factors = []
        
        data_factor = min(1.0, len(data['years']) / 12)
        factors.append(data_factor)
        
        std = np.std(data['counts'])
        mean = np.mean(data['counts'])
        cv = std / mean if mean > 0 else 1
        stability_factor = max(0, 1 - min(cv, 1))
        factors.append(stability_factor)
        
        last_seen = max(data['years'])
        years_since = target_year - last_seen
        recency_factor = np.exp(-years_since / 2.5)
        factors.append(recency_factor)
        
        years_span = max(data['years']) - min(data['years']) + 1
        appearance_rate = len(data['years']) / years_span if years_span > 0 else 1
        factors.append(appearance_rate)
        
        is_core = any(data['topic'] in CORE_TOPICS.get(s, []) for s in CORE_TOPICS)
        core_factor = 0.9 if is_core else 0.5
        factors.append(core_factor)
        
        if uncertainty > 0:
            uncertainty_factor = 1 / (1 + uncertainty)
        else:
            uncertainty_factor = 0.8
        factors.append(uncertainty_factor)
        
        if 0.3 <= prediction <= 8:
            reasonableness = 1.0
        elif prediction < 0.3:
            reasonableness = prediction / 0.3
        else:
            reasonableness = max(0.3, 1 - (prediction - 8) / 10)
        factors.append(reasonableness)
        
        weights = [1.2, 1.5, 1.8, 1.0, 1.3, 1.1, 0.9]
        confidence = np.average(factors, weights=weights[:len(factors)])
        
        return min(0.98, max(0.1, confidence))
    
    def _detect_trend(self, data):
        if len(data['counts']) < 3:
            return "Stable"
        
        recent = np.mean(data['counts'][-3:])
        older = np.mean(data['counts'][:-3]) if len(data['counts']) > 3 else np.mean(data['counts'])
        
        if older == 0:
            return "Emerging" if recent > 0 else "Stable"
        
        change_pct = ((recent - older) / older) * 100
        
        if change_pct > 30:
            return "Rising"
        elif change_pct < -30:
            return "Declining"
        else:
            return "Stable"
    
    def _normalize_to_nest(self, predictions, num_papers=1):
        """Normalize predictions to match NEST structure with blending.
        
        Improved version that preserves some of the raw ML predictions
        instead of forcing everything to fixed quotas.
        
        Args:
            predictions: List of topic predictions
            num_papers: Number of papers to predict for (default 1)
        """
        # Blend factor: 0.4 means 40% raw ML, 60% normalized to quota
        # Increased from 0.25 to allow more ML-driven predictions
        NORMALIZATION_BLEND = 0.85
        
        target_total = TOTAL_NEST_QUESTIONS * num_papers
        
        filtered = [p for p in predictions 
                   if p['confidence'] >= MIN_CONFIDENCE_THRESHOLD * 100]
        
        print(f"      Filtered {len(predictions) - len(filtered)} low-confidence topics")
        
        subject_groups = defaultdict(list)
        for p in filtered:
            subject_groups[p['subject']].append(p)
        
        normalized = []
        subject_totals = {}  # Track totals for diagnostics
        
        # Store raw predictions for blending
        raw_predictions_by_key = {}
        for p in filtered:
            key = f"{p['subject']}::{p['topic']}"
            raw_predictions_by_key[key] = p['predicted_count']
        
        for subject, base_target in SECTION_DISTRIBUTION.items():
            target_count = base_target * num_papers  # Scale for multi-paper
            topics = subject_groups.get(subject, [])
            
            if not topics:
                # Generate predictions from core topics when no historical data
                core_topics = CORE_TOPICS.get(subject, [subject])[:5]
                num_core = min(len(core_topics), max(1, int(target_count)))
                for ct in core_topics[:num_core]:
                    normalized.append({
                        'subject': subject,
                        'topic': ct,
                        'predicted_count': target_count / num_core,
                        'confidence': 45.0,  # Slightly higher confidence for core topics
                        'raw_confidence': 45.0,
                        'trend': 'Core',
                        'is_core': '✓'
                    })
                subject_totals[subject] = target_count
                continue
            
            # Sort by confidence for priority allocation
            topics.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate weighted predictions with confidence boost for high-confidence topics
            total_weighted = 0
            for p in topics:
                # Boost weight for high confidence predictions
                conf_factor = (p['confidence'] / 100) ** 0.8  # Softer power curve
                p['_weight'] = p['predicted_count'] * conf_factor
                total_weighted += p['_weight']
            
            if total_weighted == 0:
                # Equal distribution if no weighted data
                for p in topics:
                    p['predicted_count'] = target_count / len(topics)
            else:
                # Distribute target_count proportionally by weight
                for p in topics:
                    normalized_pred = (p['_weight'] / total_weighted) * target_count
                    
                    # Blend with raw prediction to preserve ML insight
                    key = f"{p['subject']}::{p['topic']}"
                    raw_pred = raw_predictions_by_key.get(key, normalized_pred)
                    
                    # Blend: preserve some raw signal while conforming to quotas
                    p['predicted_count'] = (
                        NORMALIZATION_BLEND * raw_pred + 
                        (1 - NORMALIZATION_BLEND) * normalized_pred
                    )
                    # Clean up temp field
                    del p['_weight']
            
            # Re-scale to ensure this subject hits its target exactly
            # (after any adjustments that may have changed the sum)
            current_sum = sum(p['predicted_count'] for p in topics)
            if current_sum > 0 and abs(current_sum - target_count) > 0.01:
                scale_factor = target_count / current_sum
                for p in topics:
                    p['predicted_count'] *= scale_factor
            
            normalized.extend(topics)
            subject_totals[subject] = sum(p['predicted_count'] for p in topics)
        
        # Final scaling to exact target
        total = sum(p['predicted_count'] for p in normalized)
        if total > 0 and abs(total - target_total) > 0.1:
            scale = target_total / total
            for p in normalized:
                p['predicted_count'] = round(p['predicted_count'] * scale, 2)
        
        # Sort by confidence and prediction count
        normalized.sort(key=lambda x: (x['confidence'], x['predicted_count']), reverse=True)
        
        final_total = sum(p['predicted_count'] for p in normalized)
        print(f"      ✓ Normalized: {len(normalized)} topics → {final_total:.1f} questions")
        print(f"      ℹ️  Blend factor: {NORMALIZATION_BLEND:.0%} raw + {1-NORMALIZATION_BLEND:.0%} normalized")
        
        # Diagnostic output for subject distribution
        if DIAGNOSTIC_MODE:
            print(f"      📊 Subject allocation:")
            for subj in sorted(SECTION_DISTRIBUTION.keys()):
                actual = sum(p['predicted_count'] for p in normalized if p['subject'] == subj)
                target = SECTION_DISTRIBUTION[subj] * num_papers
                diff = actual - target
                status = "✅" if abs(diff) <= 1 else "⚠️"
                print(f"         {status} {subj[:20]:<20s}: {actual:5.1f} (target: {target})")
        
        return normalized
    
    def _fallback_train(self, topic_data):
        self.all_topics_data = topic_data
        return {"MAE": 0, "RMSE": 0, "R2": 0}
    
    def _fallback_predict_single(self, data):
        weights = [get_era_weight(y) for y in data['years']]
        return np.average(data['counts'], weights=weights)

# ==========================================
# COMPREHENSIVE VALIDATION
# ==========================================

class EliteValidator:
    
    def __init__(self, predictor, processed_data):
        self.predictor = predictor
        self.processed_data = processed_data
        self.validation_results = {}
    
    def walk_forward_validation(self, test_years=[2024, 2023, 2022, 2020]):
        print("\n🔬 Walk-Forward Historical Validation...")
        
        results = []
        
        for test_year in sorted(test_years):
            # Count papers and questions per paper for this year
            paper_subjects = defaultdict(lambda: defaultdict(int))
            paper_topics = defaultdict(lambda: defaultdict(int))
            paper_micro_topics = defaultdict(lambda: defaultdict(int))
            for q in self.processed_data:
                if q['year'] == test_year:
                    paper_id = q.get('paper_id', 'default')
                    subj = q['subject']
                    topic = q.get('topic', 'Unknown')
                    micro = q.get('micro_topic', 'Unknown')
                    
                    paper_subjects[paper_id][subj] += 1
                    paper_topics[paper_id][f"{subj}::{topic}"] += 1
                    paper_micro_topics[paper_id][f"{subj}::{topic}::{micro}"] += 1
            
            num_papers = len(paper_subjects)
            if num_papers == 0:
                print(f"   ⚠️  Skipping {test_year} (no data)")
                continue
            
            # For multi-paper years, average the counts per paper
            # This allows fair comparison with 65-question predictions
            actual_subject = defaultdict(float)
            actual_topic = defaultdict(float)
            actual_micro_topic = defaultdict(float)
            for paper_id in paper_subjects.keys():
                for subj, count in paper_subjects[paper_id].items():
                    actual_subject[subj] += count / num_papers
                for t_key, count in paper_topics[paper_id].items():
                    actual_topic[t_key] += count / num_papers
                for m_key, count in paper_micro_topics[paper_id].items():
                    actual_micro_topic[m_key] += count / num_papers
            
            # Round to nearest integer for display
            actual_subject = {s: round(c) for s, c in actual_subject.items()}
            
            total_actual = sum(actual_subject.values())
            
            if DIAGNOSTIC_MODE and num_papers > 1:
                print(f"   📊 {test_year}: {num_papers} papers detected, averaging to ~{total_actual} questions")
            if total_actual < 30:
                print(f"   ⚠️  Skipping {test_year} (only {total_actual} questions)")
                continue
            
            train_data = [q for q in self.processed_data if q['year'] < test_year]
            
            if len(train_data) < 100:
                print(f"   ⚠️  Skipping {test_year} (insufficient training data)")
                continue
            
            topic_data, micro_topic_data = self._build_topic_data(train_data)
            
            predictions = []
            for key, data in topic_data.items():
                if len(data['years']) < 2:
                    continue
                weights = [get_era_weight(y) for y in data['years']]
                counts_array = np.array(data['counts'])
                pred = np.average(counts_array, weights=weights)
                raw_conf = self.predictor._calculate_elite_confidence(
                    data, pred, 0, test_year
                )
                predictions.append({
                    'subject': data['subject'],
                    'topic': data['topic'],
                    'predicted_count': max(0, pred),
                    'confidence': raw_conf * 100
                })
            
            normalized = self.predictor._normalize_to_nest(predictions)
            
            pred_subject = defaultdict(float)
            for p in normalized:
                pred_subject[p['subject']] += p['predicted_count']
            
            errors = []
            for subj in SECTION_DISTRIBUTION:
                pred = pred_subject.get(subj, 0)
                actual = actual_subject.get(subj, 0)
                error = abs(pred - actual)
                errors.append({
                    'subject': subj,
                    'predicted': round(pred, 2),
                    'actual': actual,
                    'error': error
                })
                
                # Feed calibration data
                if actual > 0:
                    # Find predictions for this subject
                    subj_preds = [p for p in normalized if p['subject'] == subj]
                    for sp in subj_preds:
                        # Accuracy: inverse of relative error
                        accuracy = 1.0 - min(1.0, error / max(1, actual))
                        self.predictor.calibrator.add_prediction(
                            sp.get('raw_confidence', sp['confidence']) / 100,
                            accuracy
                        )
            
            topic_errors = []
            predicted_topic_keys = set()
            for p in normalized:
                subj = p['subject']
                topic = p['topic']
                topic_key = f"{subj}::{topic}"
                predicted_topic_keys.add(topic_key)
                
                pred = p['predicted_count']
                actual = actual_topic.get(topic_key, 0.0)
                error = abs(pred - actual)
                topic_errors.append({
                    'subject': subj,
                    'topic': topic,
                    'predicted': round(pred, 2),
                    'actual': round(actual, 2),
                    'error': error
                })
            
            for topic_key, actual in actual_topic.items():
                if topic_key not in predicted_topic_keys:
                    parts = topic_key.split("::")
                    subj = parts[0]
                    topic = parts[1] if len(parts) > 1 else 'Unknown'
                    topic_errors.append({
                        'subject': subj,
                        'topic': topic,
                        'predicted': 0.0,
                        'actual': round(actual, 2),
                        'error': actual
                    })
            
            # Predict Micro-Topics
            original_micro_data = getattr(self.predictor, 'all_micro_topics_data', None)
            self.predictor.all_micro_topics_data = micro_topic_data
            
            import io, sys
            # Suppress print output from predict_micro_topics during validation loop
            _stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                micro_preds = self.predictor.predict_micro_topics(normalized, test_year)
            finally:
                sys.stdout = _stdout
            
            if original_micro_data is not None:
                self.predictor.all_micro_topics_data = original_micro_data
                
            micro_errors = []
            predicted_micro_keys = set()
            for mp in micro_preds:
                subj = mp['subject']
                topic = mp['topic']
                micro = mp['micro_topic']
                micro_key = f"{subj}::{topic}::{micro}"
                predicted_micro_keys.add(micro_key)
                
                pred = mp.get('predicted_count', 0.0)
                actual = actual_micro_topic.get(micro_key, 0.0)
                error = abs(pred - actual)
                micro_errors.append({
                    'subject': subj,
                    'topic': topic,
                    'micro_topic': micro,
                    'predicted': round(pred, 2),
                    'actual': round(actual, 2),
                    'error': error
                })
                
            for micro_key, actual in actual_micro_topic.items():
                if micro_key not in predicted_micro_keys:
                    parts = micro_key.split("::")
                    subj = parts[0]
                    topic = parts[1] if len(parts) > 1 else 'Unknown'
                    micro = parts[2] if len(parts) > 2 else 'Unknown'
                    micro_errors.append({
                        'subject': subj,
                        'topic': topic,
                        'micro_topic': micro,
                        'predicted': 0.0,
                        'actual': round(actual, 2),
                        'error': actual
                    })
            
            mae = np.mean([e['error'] for e in errors]) if errors else 0.0
            topic_mae = np.mean([e['error'] for e in topic_errors]) if topic_errors else 0.0
            micro_mae = np.mean([e['error'] for e in micro_errors]) if micro_errors else 0.0
            
            results.append({
                'year': test_year,
                'mae': round(mae, 3),
                'topic_mae': round(topic_mae, 3),
                'micro_topic_mae': round(micro_mae, 3),
                'total_pred': round(sum(pred_subject.values()), 1),
                'total_actual': sum(actual_subject.values()),
                'errors': errors,
                'topic_errors': topic_errors,
                'micro_errors': micro_errors
            })
            
            print(f"   {test_year}: Subject MAE={mae:.3f} | Topic MAE={topic_mae:.3f} | Micro MAE={micro_mae:.3f} | Pred={sum(pred_subject.values()):.0f} vs Actual={sum(actual_subject.values())}")

        
        # Calibrate after collecting all data
        self.predictor.calibrator.calibrate()
        
        self.validation_results['walk_forward'] = results
        self.validation_results['calibration'] = self.predictor.calibrator.get_calibration_metrics()
        
        return results
    
    def _build_topic_data(self, questions):
        td = {}
        md = {}
        
        for q in questions:
            key = f"{q['subject']}::{q['topic']}"
            if key not in td:
                td[key] = {
                    'years': [],
                    'counts': [],
                    'year_counts': {},
                    'subject': '',
                    'topic': ''
                }
            td[key]['years'].append(q['year'])
            td[key]['year_counts'][q['year']] = td[key]['year_counts'].get(q['year'], 0) + 1
            td[key]['subject'] = q['subject']
            td[key]['topic'] = q['topic']
            
            micro = q.get('micro_topic', q['topic'])
            m_key = f"{q['subject']}::{q['topic']}::{micro}"
            if m_key not in md:
                md[m_key] = {
                    'years': [],
                    'counts': [],
                    'year_counts': {},
                    'subject': '',
                    'topic': '',
                    'micro_topic': ''
                }
            md[m_key]['years'].append(q['year'])
            md[m_key]['year_counts'][q['year']] = md[m_key]['year_counts'].get(q['year'], 0) + 1
            md[m_key]['subject'] = q['subject']
            md[m_key]['topic'] = q['topic']
            md[m_key]['micro_topic'] = micro
        
        for key in td:
            years = sorted(set(td[key]['years']))
            td[key]['years'] = years
            td[key]['counts'] = [td[key]['year_counts'][y] for y in years]
            
        for key in md:
            years = sorted(set(md[key]['years']))
            md[key]['years'] = years
            md[key]['counts'] = [md[key]['year_counts'][y] for y in years]
        
        return td, md
    
    def generate_report(self):
        print("\n" + "="*80)
        print("📋 ELITE VALIDATION REPORT")
        print("="*80)
        
        if 'walk_forward' in self.validation_results:
            results = self.validation_results['walk_forward']
            if results:
                avg_mae = np.mean([r['mae'] for r in results])
                avg_topic_mae = np.mean([r.get('topic_mae', 0) for r in results])
                avg_micro_mae = np.mean([r.get('micro_topic_mae', 0) for r in results])
                
                print(f"\n🔬 Walk-Forward Validation:")
                print(f"   Average Subject MAE: {avg_mae:.3f}")
                print(f"   Average Topic MAE:   {avg_topic_mae:.3f}")
                print(f"   Average Micro MAE:   {avg_micro_mae:.3f}")
                
                for r in results:
                    print(f"\n   {r['year']}:")
                    print(f"      Subject MAE: {r['mae']:.3f} | Topic MAE: {r.get('topic_mae', 0):.3f} | Micro MAE: {r.get('micro_topic_mae', 0):.3f}")
                    print(f"      Total: Pred={r['total_pred']:.0f}, Actual={r['total_actual']}")
                    print(f"      Top Subject Errors:")
                    top_errors = sorted(r['errors'], key=lambda x: x['error'], reverse=True)[:3]
                    for e in top_errors:
                        print(f"         {e['subject'][:25]:25s}: pred={e['predicted']:5.1f}, actual={e['actual']:2d}, err={e['error']:.1f}")
                    
                    if 'topic_errors' in r:
                        print(f"      Top Topic Errors:")
                        top_topic_errors = sorted(r['topic_errors'], key=lambda x: x['error'], reverse=True)[:3]
                        for e in top_topic_errors:
                            print(f"         {e['topic'][:25]:25s}: pred={e['predicted']:5.1f}, actual={e['actual']:5.1f}, err={e['error']:.1f}")
                            
                    if 'micro_errors' in r:
                        print(f"      Top Micro-Topic Errors:")
                        top_micro_errors = sorted(r['micro_errors'], key=lambda x: x['error'], reverse=True)[:3]
                        for e in top_micro_errors:
                            print(f"         {e['micro_topic'][:25]:25s}: pred={e['predicted']:5.1f}, actual={e['actual']:5.1f}, err={e['error']:.1f}")
        
        if 'calibration' in self.validation_results:
            cal = self.validation_results['calibration']
            if cal:
                print(f"\n📊 Confidence Calibration:")
                print(f"   Calibration Error (ECE): {cal.get('calibration_error', 0):.4f}")
                print(f"   Bin Accuracies:")
                for bin_range, acc in cal.get('bin_accuracies', {}).items():
                    count = cal['bin_counts'].get(bin_range, 0)
                    print(f"      {bin_range}: {acc:.3f} ({count} samples)")
        
        print("\n" + "="*80)
        return self.validation_results

# ==========================================
# DIAGNOSTIC FUNCTIONS
# ==========================================

def diagnose_data_issues(file_path="all_merged_cleaned.json"):
    """Run comprehensive diagnostics on the data file."""
    print("\n" + "="*80)
    print("🔍 DATA DIAGNOSTICS")
    print("="*80)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return
    
    print(f"\n📊 Total records: {len(data)}")
    
    # Check field presence
    fields = defaultdict(int)
    for q in data:
        for key in q.keys():
            fields[key] += 1
    
    print(f"\n📋 Field coverage:")
    for field, count in sorted(fields.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        status = "✅" if pct > 90 else ("⚠️" if pct > 50 else "❌")
        print(f"   {status} {field:<25s}: {count:5d} ({pct:5.1f}%)")
    
    # Check subject values
    subjects = Counter(q.get('subject') or 'MISSING' for q in data)
    print(f"\n📋 Subject values (top 20):")
    for subj, count in subjects.most_common(20):
        normalized = normalize_subject(subj)
        status = "✅" if normalized else "❌"
        mapped = f" → {normalized}" if normalized else " (UNMAPPED)"
        print(f"   {status} {count:4d}× '{subj}'{mapped}")
    
    # Check year distribution
    years = Counter(q.get('year', 'MISSING') for q in data)
    print(f"\n📅 Year distribution:")
    for year in sorted(years.keys()):
        print(f"   {year}: {years[year]} questions")
    
    # Check papers per year
    print(f"\n📋 Papers per year:")
    year_papers = defaultdict(set)
    for q in data:
        year = q.get('year')
        paper = q.get('shift', 'Shift 1')
        if year:
            year_papers[year].add(paper)
    
    for year in sorted(year_papers.keys()):
        papers = year_papers[year]
        print(f"   {year}: {len(papers)} paper(s) - {papers}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("=" * 80)
    print("🎯 NEST 2027 ELITE PREDICTION SYSTEM v3.2 (with Calibration)")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target: {TOTAL_NEST_QUESTIONS} questions, {len(SECTION_DISTRIBUTION)} subjects")
    print(f"🧠 Features: 55 advanced features")
    print(f"🤖 Models: Multi-layer ensemble with confidence calibration")
    print("=" * 80)
    
    # Run diagnostics first
    # Uncomment the line below to run full diagnostics
    # diagnose_data_issues("all_merged_cleaned.json")
    
    processed_data = load_and_process_data("all_merged_cleaned.json", verbose=True)
    
    if len(processed_data) < 100:
        print("❌ Insufficient data")
        return
    
    predictor = EliteNESTPredictor()
    metrics = predictor.train(processed_data)
    
    # Validation with calibration (uses walk-forward: trains on years < test_year)
    validator = EliteValidator(predictor, processed_data)
    validator.walk_forward_validation([2024, 2023, 2022, 2020, 2019])
    validation_report = validator.generate_report()
    
    # RETRAIN on FULL 2011-2026 data for final 2027 prediction
    print("\n" + "="*80)
    print("🔄 RETRAINING on FULL 2007-2024 data for 2025 prediction...")
    print("="*80)
    predictor_final = EliteNESTPredictor()
    metrics_final = predictor_final.train(processed_data)  # Now includes 2024-2026
    print(f"   ✅ Retrained on {len(processed_data)} questions (2007-2024)")
    
    # Generate final predictions with calibrated confidence
    predictions = predictor_final.predict(2025)
    
    # Generate micro-topic predictions
    micro_predictions = predictor_final.predict_micro_topics(predictions, 2025)
    
    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETE")
    print("="*80)
    
    subject_totals = defaultdict(float)
    for p in predictions:
        subject_totals[p['subject']] += p['predicted_count']
    
    print(f"\n📊 Subject Distribution:")
    print(f"   {'Subject':<30s} {'Predicted':>10s} {'Target':>8s} {'Diff':>8s}")
    print(f"   {'-'*60}")
    
    total_diff = 0
    for subject in SECTION_DISTRIBUTION:
        pred = subject_totals.get(subject, 0)
        target = SECTION_DISTRIBUTION[subject]
        diff = pred - target
        total_diff += abs(diff)
        status = "✅" if abs(diff) <= 1 else "⚠️"
        print(f"   {status} {subject:<28s} {pred:>8.1f}   {target:>5d}    {diff:>+6.1f}")
    
    print(f"\n   Total Absolute Deviation: {total_diff:.1f}")
    
    print(f"\n🔝 Top 15 Predictions:")
    for i, p in enumerate(predictions[:15], 1):
        conf_bar = "█" * int(p['confidence'] / 10) + "░" * (10 - int(p['confidence'] / 10))
        cal_indicator = f"[Cal: {p.get('raw_confidence', p['confidence']):.0f}→{p['confidence']:.0f}]" if 'raw_confidence' in p else ""
        print(f"   {i:2}. {p['subject'][:20]:20s} | {p['topic'][:26]:26s} → {p['predicted_count']:5.2f}q | {conf_bar} {p['confidence']:5.1f}% {cal_indicator}")
    
    # Print top micro-topic predictions
    if micro_predictions:
        print(f"\n🔬 Top 15 Micro-Topic Predictions:")
        for i, m in enumerate(micro_predictions[:15], 1):
            conf_bar = "█" * int(m['confidence'] / 10) + "░" * (10 - int(m['confidence'] / 10))
            print(f"   {i:2}. {m['topic'][:18]:18s} → {m['micro_topic'][:28]:28s} {m['predicted_count']:5.3f}q | {conf_bar} {m['confidence']:5.1f}%")
    
    output = {
        'meta': {
            'version': '3.4 Elite NEST Adaptation with Weighted Recency Micro-Topics',
            'generated_at': datetime.now().isoformat(),
            'total_analyzed': len(processed_data),
            'features': 55,
            'models': len(predictor.base_models),
            'metrics': metrics
        },
        'predictions': predictions,
        'micro_topic_predictions': micro_predictions,
        'validation': validation_report,
        'subject_totals': dict(subject_totals)
    }
    
    with open('nest_elite_predictions_2027.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Saved: nest_elite_predictions_2027.json")
    print(f"   📊 {len(predictions)} topic predictions + {len(micro_predictions)} micro-topic predictions")
    print("="*80)

if __name__ == "__main__":
    main()