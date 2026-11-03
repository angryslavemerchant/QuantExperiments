from shapequant.alloc import budget_bits, heuristic, lagrangian


def test_lagrangian_meets_budget_and_prefers_big_gains():
    err = [[10, 4, 1], [5, 4, 3.9], [8, 2, 1.5]]
    bits = [[100, 125, 150]] * 3
    for B in (300, 350, 375, 400, 450):
        ch = lagrangian(err, bits, B)
        assert sum(bits[i][k] for i, k in enumerate(ch)) <= B
    assert lagrangian(err, bits, 350) == [1, 0, 1]  # the low-gain matrix stays small
    assert lagrangian(err, bits, 450) == [2, 2, 2]


def test_heuristic_mixes_around_the_uniform_level():
    names = ["model.layers.12.self_attn.q_proj", "model.layers.12.mlp.down_proj",
             "model.layers.0.mlp.up_proj", "model.layers.12.mlp.up_proj"]
    bits = [[100, 125, 150]] * 4
    ch = heuristic(names, bits, 500, 24)          # budget = uniform level 1
    assert sum(bits[i][k] for i, k in enumerate(ch)) <= 500
    assert ch[1] == 2 and ch[2] == 2               # important matrices upgraded
    assert ch[3] == 0                              # mid-block MLP pays first
    assert ch != [1, 1, 1, 1]


def test_budget_spec():
    bits = [[100, 125, 150], [10, 20, 30]]
    assert budget_bits(bits, (256, 1024, 4096), "uniform:1024") == 145
