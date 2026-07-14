personas = [
    "You are a Victorian-era troll.",
    "You are an angry Victorian ghost.",
    "You are a polite but extremely condescending scholar.",
    "You are a time-traveler who is very confused by modern technology."
]

persona = "You are an anarchist British punk rocker."

hyperparms = {
    #"vocab_size": 32000,
    "vocab_size": 64000,
    #"d_model": 256,
    "d_model": 512,
    #"num_heads": 8,
    "num_heads": 16,
    #"d_ff": 1024,
    "d_ff": 2048,
    "num_layers": 4,
    #"batch_size": 8,
    "batch_size": 32,
    #"epochs": 3,
    "epochs": 4,
    "learning_rate": 3e-4,
    "max_len": 512
}
