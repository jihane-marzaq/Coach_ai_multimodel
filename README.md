la partie nlp de coachmodel aura consiste a : 
Le modèle fonctionne comme un système intelligent d’évaluation de qualité de texte basé sur le NLP et le Machine Learning.
Lorsqu’un utilisateur saisit un texte, celui-ci passe d’abord par une phase de prétraitement avec [spaCy]
afin de nettoyer le contenu, segmenter les phrases et extraire les informations linguistiques importantes.
Ensuite, un système de scoring automatique calcule plusieurs indicateurs objectifs comme la lisibilité du texte (Flesch Reading Ease), 
la diversité lexicale (richesse du vocabulaire) et la qualité grammaticale grâce à [LanguageTool]. 
Le texte est ensuite transformé en représentation sémantique dense à l’aide de [Sentence-BERT]
qui convertit le sens global du texte en vecteurs numériques compréhensibles par le modèle de Machine Learning.
Ces embeddings sont utilisés pour entraîner un modèle de régression Random Forest avec [scikit-learn]
afin de prédire automatiquement un score de qualité compris entre 0 et 10. Enfin, le système peut générer un feedback intelligent expliquant les points forts et les faiblesses du texte pour aider l’utilisateur à améliorer sa communication orale ou écrite.
