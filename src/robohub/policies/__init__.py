from robohub.policies.act import ACTPolicy
from robohub.policies.base import Policy
from robohub.policies.fake_policy import FakePolicy
from robohub.policies.pi05 import PI05Policy

POLICY_REGISTRY = {"act": ACTPolicy, "fake_policy": FakePolicy, "pi05": PI05Policy}

__all__ = ["ACTPolicy", "FakePolicy", "PI05Policy", "Policy", "POLICY_REGISTRY"]
