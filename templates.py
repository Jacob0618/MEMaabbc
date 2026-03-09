
seq_templates = [
    'Given the following purchase history of user_{}: item_{}, predict next possible item to be purchased by the user.',
    'I find the purchase history list of user_{}: item_{}. I wonder what is the next item to recommend to the user. Can you help me decide?',
    'Here is the purchase history list of user_{}: item_{}. Try to recommend next item to the user.',
    'Given the following purchase history of user_{}: item_{}, predict next possible item for the user.',
    'Based on the purchase history of user_{}: item_{}, can you decide the next item likely to be purchased by the user?',
    'Here is the purchase history of user_{}: item_{}. What to recommend next for the user?',
    'According to the purchase history of user_{}: item_{}, can you recommend the next possible item to the user?',
    'user_{} item_{}',
]

# 3-slot templates: user_id, history_items, candidate_items
topn_templates = [
    'Given user_{} recent history: item_{}, recommend from candidates: item_{}',
    'user_{} has recently interacted with: item_{}. Choose the best from: item_{}',
    'Based on user_{} purchases: item_{}, select the best item from: item_{}',
    'For user_{} with history: item_{}, pick the best recommendation from: item_{}',
    'user_{} item_{} item_{}',
]

# 2-slot fallback for samples with no history
topn_templates_no_history = [
    'Which item of the following to recommend for user_{}? item_{}',
    'Choose the best item from the candidates to recommend for user_{}? item_{}',
    'We want to make recommendation for user_{}. Select the best item from these candidates: item_{}',
    'user_{} item_{}',
]

exp_templates = [
    'Generate an explanation for user_{} about this product: item_{}',
    'Can you help generate an explanation of user_{} for item_{}?',
    'Help user_{} generate an explanation about this product: item_{}',
    'Generate user_{}\'s purchase explanation about item_{}',
    'Help user_{} generate an explanation for item_{}',
    'Can you help generate an explanation for user_{} about the product: item_{}',
    'Write an explanation for user_{} about item_{}',
    'Generate an explanation for user_{} about item_{}',
    'user_{} item_{}',
]
