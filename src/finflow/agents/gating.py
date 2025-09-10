class Gating:
    def __init__(self, z_to_lambda_scale: float = 1.0, z_to_entropy_scale: float = 0.5):
        self.z2lam = z_to_lambda_scale
        self.z2ent = z_to_entropy_scale

    def modulate(self, z: float):
        # returns (lambda_scale, entropy_scale)
        return (1.0 + self.z2lam*z, max(0.1, 1.0 - self.z2ent*z))
