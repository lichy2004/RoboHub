from robohub.policies.base import Policy
from robohub.policies.fake_policy import FakePolicy
from robohub.policies.act import ACTPolicy

POLICY_REGISTRY = {"act": ACTPolicy, "fake_policy": FakePolicy}

__all__ = ["ACTPolicy", "FakePolicy", "Policy", "POLICY_REGISTRY"]
