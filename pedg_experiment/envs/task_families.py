"""
Synthetic Task Environment for PEDG Experiment
10 task families with controllable ground truth, public/private evidence
"""
import random
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TaskInstance:
    """Single task instance with ground truth"""
    family: str
    instance_id: str
    z: str  # Ground truth: "Good" or "Bad"
    e_pub: Dict[str, Any]  # Public evidence
    e_priv: Dict[str, Any]  # Private evidence
    ground_truth_decision: str  # APPROVE or REJECT
    key_private_fields: List[str]  # Which private fields are critical
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskFamily(ABC):
    """Base class for a task family"""
    name: str = "base"
    description: str = ""
    
    @abstractmethod
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        pass
    
    @abstractmethod
    def ground_truth_rule(self, e_pub: Dict, e_priv: Dict) -> str:
        """Given evidence, return APPROVE or REJECT"""
        pass


# ============ Task Family 1: Project Approval ============
class ProjectApproval(TaskFamily):
    name = "project_approval"
    description = "Approve or reject project funding requests"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        score = e_pub.get("budget_justification_score", 5) * 0.5
        score += e_priv.get("team_experience_score", 5) * 0.3
        score += e_priv.get("risk_assessment_score", 5) * 0.2
        return "APPROVE" if score >= 4.5 else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        # 60% Good, 40% Bad to simulate realistic approval rates
        is_good = rng.random() < 0.6
        
        if is_good:
            e_pub = {
                "project_name": rng.choice(["AI Platform", "Cloud Migration", "Data Pipeline", "Mobile App"]),
                "budget_justification_score": rng.randint(6, 10),
                "alignment_with_strategy": rng.choice(["high", "medium"]),
                "requested_amount_usd": rng.randint(50000, 500000),
            }
            e_priv = {
                "team_experience_score": rng.randint(6, 10),
                "risk_assessment_score": rng.randint(6, 10),
                "previous_project_success_rate": rng.randint(70, 100),
                "has_contingency_plan": True,
            }
            key_fields = ["team_experience_score", "risk_assessment_score"]
        else:
            e_pub = {
                "project_name": rng.choice(["AI Platform", "Cloud Migration", "Data Pipeline", "Mobile App"]),
                "budget_justification_score": rng.randint(3, 7),
                "alignment_with_strategy": rng.choice(["medium", "low"]),
                "requested_amount_usd": rng.randint(50000, 500000),
            }
            e_priv = {
                "team_experience_score": rng.randint(1, 5),
                "risk_assessment_score": rng.randint(1, 5),
                "previous_project_success_rate": rng.randint(10, 60),
                "has_contingency_plan": False,
            }
            key_fields = ["team_experience_score", "risk_assessment_score", "has_contingency_plan"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
            metadata={"requested_amount": e_pub["requested_amount_usd"]}
        )


# ============ Task Family 2: Loan Approval ============
class LoanApproval(TaskFamily):
    name = "loan_approval"
    description = "Approve or reject loan applications"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        score = e_pub.get("credit_score_normalized", 0.5) * 40
        score += e_priv.get("debt_to_income_ratio_adj", 0.3) * 30
        score += e_priv.get("employment_stability_years", 3) * 10
        score += e_priv.get("collateral_quality", 0.5) * 20
        return "APPROVE" if score >= 65 else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.55
        
        if is_good:
            e_pub = {
                "applicant_name": f"Applicant_{instance_id}",
                "credit_score_normalized": rng.uniform(0.7, 0.95),
                "loan_amount_requested": rng.randint(10000, 100000),
                "loan_purpose": rng.choice(["home_improvement", "debt_consolidation", "business"]),
            }
            e_priv = {
                "debt_to_income_ratio_adj": rng.uniform(0.15, 0.35),
                "employment_stability_years": rng.randint(3, 15),
                "collateral_quality": rng.uniform(0.7, 1.0),
                "missed_payments_last_2yrs": 0,
            }
            key_fields = ["debt_to_income_ratio_adj", "employment_stability_years", "missed_payments_last_2yrs"]
        else:
            e_pub = {
                "applicant_name": f"Applicant_{instance_id}",
                "credit_score_normalized": rng.uniform(0.3, 0.7),
                "loan_amount_requested": rng.randint(10000, 100000),
                "loan_purpose": rng.choice(["home_improvement", "debt_consolidation", "business"]),
            }
            e_priv = {
                "debt_to_income_ratio_adj": rng.uniform(0.4, 0.7),
                "employment_stability_years": rng.randint(0, 2),
                "collateral_quality": rng.uniform(0.2, 0.6),
                "missed_payments_last_2yrs": rng.randint(1, 5),
            }
            key_fields = ["debt_to_income_ratio_adj", "missed_payments_last_2yrs", "employment_stability_years"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 3: Experiment Replication Review ============
class ExperimentReplication(TaskFamily):
    name = "experiment_replication"
    description = "Review claims of successful experiment replication"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        # Need BOTH high methodology AND successful replication outcome
        methodology_ok = e_pub.get("methodology_score", 5) >= 6
        replication_ok = e_priv.get("replication_outcome_score", 5) >= 7
        return "APPROVE" if (methodology_ok and replication_ok) else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.45
        
        if is_good:
            e_pub = {
                "paper_title": f"Study on Topic {instance_id}",
                "original_study_journal": rng.choice(["Nature", "Science", "Cell", "NEJM"]),
                "methodology_score": rng.randint(7, 10),
                "sample_size_claimed": rng.randint(100, 1000),
            }
            e_priv = {
                "replication_outcome_score": rng.randint(7, 10),
                "p_value_replicated": rng.uniform(0.001, 0.04),
                "effect_size_similarity": rng.uniform(0.8, 1.0),
                "replication_lab_reputation": rng.choice(["top_10", "top_50", "established"]),
            }
            key_fields = ["replication_outcome_score", "p_value_replicated"]
        else:
            # Two failure modes: bad methodology OR bad replication outcome
            if rng.random() < 0.5:
                e_pub = {
                    "paper_title": f"Study on Topic {instance_id}",
                    "original_study_journal": rng.choice(["Nature", "Science", "Cell", "NEJM"]),
                    "methodology_score": rng.randint(3, 6),
                    "sample_size_claimed": rng.randint(50, 200),
                }
                e_priv = {
                    "replication_outcome_score": rng.randint(1, 10),
                    "p_value_replicated": rng.uniform(0.001, 0.5),
                    "effect_size_similarity": rng.uniform(0.0, 1.0),
                    "replication_lab_reputation": rng.choice(["top_10", "top_50", "established"]),
                }
                key_fields = ["methodology_score"]
            else:
                e_pub = {
                    "paper_title": f"Study on Topic {instance_id}",
                    "original_study_journal": rng.choice(["Nature", "Science", "Cell", "NEJM"]),
                    "methodology_score": rng.randint(7, 10),
                    "sample_size_claimed": rng.randint(100, 1000),
                }
                e_priv = {
                    "replication_outcome_score": rng.randint(1, 6),
                    "p_value_replicated": rng.uniform(0.05, 0.9),
                    "effect_size_similarity": rng.uniform(0.1, 0.5),
                    "replication_lab_reputation": rng.choice(["unknown", "new"]),
                }
                key_fields = ["replication_outcome_score", "effect_size_similarity"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 4: Security Vulnerability Triage ============
class SecurityTriage(TaskFamily):
    name = "security_triage"
    description = "Triage security vulnerability reports"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        severity_pub = e_pub.get("reported_severity", 5)
        exploitability_priv = e_priv.get("exploitability_score", 5)
        has_patch = e_priv.get("has_existing_patch", False)
        
        # Critical if high severity AND high exploitability AND no patch
        if severity_pub >= 8 and exploitability_priv >= 7 and not has_patch:
            return "APPROVE"  # Approve for immediate patching
        elif severity_pub <= 3:
            return "REJECT"  # Low severity, reject
        elif has_patch and exploitability_priv <= 3:
            return "REJECT"
        return "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.35  # 35% critical enough for immediate action
        
        if is_good:
            e_pub = {
                "vulnerability_id": f"CVE-2024-{1000 + ep_idx}",
                "reported_severity": rng.randint(8, 10),
                "affected_component": rng.choice(["auth_module", "payment_gateway", "API_layer", "database"]),
                "public_exploit_available": False,
            }
            e_priv = {
                "exploitability_score": rng.randint(7, 10),
                "has_existing_patch": False,
                "days_to_exploit_if_unpatched": rng.randint(1, 14),
                "sensitive_data_exposure": rng.choice(["PII", "financial", "health", "none"]),
            }
            key_fields = ["exploitability_score", "has_existing_patch", "days_to_exploit_if_unpatched"]
        else:
            e_pub = {
                "vulnerability_id": f"CVE-2024-{1000 + ep_idx}",
                "reported_severity": rng.randint(1, 8),
                "affected_component": rng.choice(["auth_module", "payment_gateway", "API_layer", "database"]),
                "public_exploit_available": rng.choice([True, False]),
            }
            e_priv = {
                "exploitability_score": rng.randint(1, 7),
                "has_existing_patch": rng.choice([True, False]),
                "days_to_exploit_if_unpatched": rng.randint(30, 365),
                "sensitive_data_exposure": rng.choice(["none", "none", "none"]),
            }
            key_fields = ["has_existing_patch", "exploitability_score"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 5: Supplier Admission ============
class SupplierAdmission(TaskFamily):
    name = "supplier_admission"
    description = "Admit new suppliers to approved vendor list"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        cost_competitive = e_pub.get("price_competitiveness", 5) >= 7
        quality_ok = e_priv.get("quality_cert_score", 5) >= 8
        reliable = e_priv.get("on_time_delivery_rate", 90) >= 85
        return "APPROVE" if (cost_competitive and quality_ok and reliable) else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.5
        
        if is_good:
            e_pub = {
                "supplier_name": f"Vendor_{instance_id}",
                "price_competitiveness": rng.randint(7, 10),
                "location": rng.choice(["domestic", "nearshore", "offshore"]),
                "years_in_business": rng.randint(5, 30),
            }
            e_priv = {
                "quality_cert_score": rng.randint(8, 10),
                "on_time_delivery_rate": rng.randint(88, 100),
                "financial_stability_rating": rng.randint(4, 5),
                "environmental_compliance": True,
            }
            key_fields = ["quality_cert_score", "on_time_delivery_rate", "financial_stability_rating"]
        else:
            e_pub = {
                "supplier_name": f"Vendor_{instance_id}",
                "price_competitiveness": rng.randint(3, 8),
                "location": rng.choice(["domestic", "nearshore", "offshore"]),
                "years_in_business": rng.randint(1, 10),
            }
            e_priv = {
                "quality_cert_score": rng.randint(3, 8),
                "on_time_delivery_rate": rng.randint(50, 90),
                "financial_stability_rating": rng.randint(1, 3),
                "environmental_compliance": rng.choice([True, False]),
            }
            key_fields = ["quality_cert_score", "on_time_delivery_rate", "financial_stability_rating"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 6: Procurement Approval ============
class ProcurementApproval(TaskFamily):
    name = "procurement_approval"
    description = "Approve procurement requests above budget threshold"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        amount = e_pub.get("amount_usd", 50000)
        budget_available = e_priv.get("budget_available_usd", 0)
        urgency = e_priv.get("business_urgency", 3)
        price_match = e_priv.get("price_match_to_market", 0.95)
        
        if amount > budget_available * 1.5:
            return "REJECT"
        if urgency >= 8 and price_match >= 0.9:
            return "APPROVE"
        if urgency >= 5 and price_match >= 0.85:
            return "APPROVE"
        return "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.5
        
        if is_good:
            e_pub = {
                "request_id": f"PO-{instance_id}",
                "item_description": rng.choice(["server_hardware", "software_license", "consulting_service"]),
                "amount_usd": rng.randint(20000, 200000),
                "requested_by": rng.choice(["engineering", "sales", "marketing"]),
            }
            e_priv = {
                "budget_available_usd": rng.randint(100000, 500000),
                "business_urgency": rng.randint(6, 10),
                "price_match_to_market": rng.uniform(0.88, 1.0),
                "has_exemption_approved": rng.choice([True, False]),
            }
            key_fields = ["budget_available_usd", "business_urgency", "price_match_to_market"]
        else:
            e_pub = {
                "request_id": f"PO-{instance_id}",
                "item_description": rng.choice(["server_hardware", "software_license", "consulting_service"]),
                "amount_usd": rng.randint(50000, 500000),
                "requested_by": rng.choice(["engineering", "sales", "marketing"]),
            }
            e_priv = {
                "budget_available_usd": rng.randint(10000, 80000),
                "business_urgency": rng.randint(1, 6),
                "price_match_to_market": rng.uniform(0.7, 0.95),
                "has_exemption_approved": False,
            }
            key_fields = ["budget_available_usd", "price_match_to_market", "business_urgency"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 7: Release Approval ============
class ReleaseApproval(TaskFamily):
    name = "release_approval"
    description = "Approve software release to production"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        tests_pass = e_priv.get("automated_test_pass_rate", 80) >= 90
        no_critical_bugs = e_priv.get("open_critical_bugs", 2) == 0
        security_scan = e_pub.get("security_scan_result", "pass") in ["pass", "warning"]
        return "APPROVE" if (tests_pass and no_critical_bugs and security_scan) else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.55
        
        if is_good:
            e_pub = {
                "release_version": f"v2.{instance_id}.0",
                "security_scan_result": rng.choice(["pass", "pass", "warning"]),
                "deployment_target": rng.choice(["production", "staging"]),
                "change_type": rng.choice(["feature", "bugfix", "hotfix"]),
            }
            e_priv = {
                "automated_test_pass_rate": rng.randint(92, 100),
                "open_critical_bugs": 0,
                "code_review_approved": True,
                "rollback_plan_documented": True,
            }
            key_fields = ["automated_test_pass_rate", "open_critical_bugs", "code_review_approved"]
        else:
            e_pub = {
                "release_version": f"v2.{instance_id}.0",
                "security_scan_result": rng.choice(["fail", "pass", "warning"]),
                "deployment_target": rng.choice(["production", "staging"]),
                "change_type": rng.choice(["feature", "bugfix", "hotfix"]),
            }
            e_priv = {
                "automated_test_pass_rate": rng.randint(60, 92),
                "open_critical_bugs": rng.randint(0, 3),
                "code_review_approved": rng.choice([True, False]),
                "rollback_plan_documented": rng.choice([True, False]),
            }
            key_fields = ["open_critical_bugs", "automated_test_pass_rate", "code_review_approved"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 8: Dataset Quality Check ============
class DatasetQuality(TaskFamily):
    name = "dataset_quality"
    description = "Approve dataset for use in ML training pipeline"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        completeness = e_priv.get("completeness_score", 0.8) >= 0.85
        no_leakage = e_priv.get("train_test_leakage", True) == False
        balanced = e_priv.get("class_balance_ratio", 0.3) >= 0.2
        accurate = e_pub.get("label_accuracy_sample", 0.9) >= 0.88
        return "APPROVE" if (completeness and no_leakage and balanced and accurate) else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.45
        
        if is_good:
            e_pub = {
                "dataset_name": f"dataset_{instance_id}",
                "num_samples": rng.randint(10000, 500000),
                "feature_dimensions": rng.randint(10, 500),
                "label_accuracy_sample": rng.uniform(0.90, 0.99),
            }
            e_priv = {
                "completeness_score": rng.uniform(0.87, 0.99),
                "train_test_leakage": False,
                "class_balance_ratio": rng.uniform(0.3, 0.6),
                "missing_value_handling": "advanced_imputation",
            }
            key_fields = ["completeness_score", "train_test_leakage", "class_balance_ratio"]
        else:
            e_pub = {
                "dataset_name": f"dataset_{instance_id}",
                "num_samples": rng.randint(1000, 100000),
                "feature_dimensions": rng.randint(5, 300),
                "label_accuracy_sample": rng.uniform(0.7, 0.9),
            }
            e_priv = {
                "completeness_score": rng.uniform(0.5, 0.87),
                "train_test_leakage": rng.choice([True, False]),
                "class_balance_ratio": rng.uniform(0.05, 0.25),
                "missing_value_handling": rng.choice(["drop", "mean", "none"]),
            }
            key_fields = ["train_test_leakage", "completeness_score", "class_balance_ratio"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 9: Medical流程合规审查 ============
class MedicalCompliance(TaskFamily):
    name = "medical_compliance"
    description = "Review medical procedure for regulatory compliance (synthetic rules only)"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        documented = e_priv.get("has_complete_documentation", False)
        informed_consent = e_priv.get("has_informed_consent", False)
        staff_qualified = e_priv.get("staff_certification_valid", False)
        protocol_followed = e_pub.get("protocol_adherence_rate", 0.9) >= 0.85
        return "APPROVE" if (documented and informed_consent and staff_qualified and protocol_followed) else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.55
        
        if is_good:
            e_pub = {
                "procedure_id": f"PROC-{instance_id}",
                "procedure_type": rng.choice(["diagnostic", "therapeutic", "surgical"]),
                "protocol_adherence_rate": rng.uniform(0.88, 1.0),
                "department": rng.choice(["cardiology", "oncology", "radiology", "neurology"]),
            }
            e_priv = {
                "has_complete_documentation": True,
                "has_informed_consent": True,
                "staff_certification_valid": True,
                "incident_free_last_90days": True,
            }
            key_fields = ["has_complete_documentation", "has_informed_consent", "staff_certification_valid"]
        else:
            # Various failure modes
            failure_mode = rng.choice(["docs", "consent", "staff"])
            e_pub = {
                "procedure_id": f"PROC-{instance_id}",
                "procedure_type": rng.choice(["diagnostic", "therapeutic", "surgical"]),
                "protocol_adherence_rate": rng.uniform(0.7, 0.9),
                "department": rng.choice(["cardiology", "oncology", "radiology", "neurology"]),
            }
            if failure_mode == "docs":
                e_priv = {"has_complete_documentation": False, "has_informed_consent": True, "staff_certification_valid": True, "incident_free_last_90days": rng.choice([True, False])}
                key_fields = ["has_complete_documentation"]
            elif failure_mode == "consent":
                e_priv = {"has_complete_documentation": True, "has_informed_consent": False, "staff_certification_valid": True, "incident_free_last_90days": rng.choice([True, False])}
                key_fields = ["has_informed_consent"]
            else:
                e_priv = {"has_complete_documentation": True, "has_informed_consent": True, "staff_certification_valid": False, "incident_free_last_90days": rng.choice([True, False])}
                key_fields = ["staff_certification_valid"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Task Family 10: Contract Risk Screening ============
class ContractRiskScreening(TaskFamily):
    name = "contract_risk_screening"
    description = "Screen contracts for legal and financial risk"
    
    def ground_truth_rule(self, e_pub, e_priv) -> str:
        legal_review = e_priv.get("legal_review_outcome", "clean") in ["clean", "minor_issues"]
        liability_exposure = e_priv.get("liability_exposure_usd", 100000) < 500000
        auto_renewal_risk = not e_priv.get("has_auto_renewal", False)
        termination_flexibility = e_priv.get("termination_flexibility", "high") in ["high", "medium"]
        return "APPROVE" if (legal_review == "clean" and liability_exposure and auto_renewal_risk and termination_flexibility) else "REJECT"
    
    def generate(self, rng: random.Random, instance_id: str, ep_idx: int = 0) -> TaskInstance:
        is_good = rng.random() < 0.5
        
        if is_good:
            e_pub = {
                "contract_id": f"CTR-{instance_id}",
                "counterparty": rng.choice(["Acme Corp", "Global LLC", "TechStart Inc"]),
                "contract_value_usd": rng.randint(50000, 2000000),
                "duration_months": rng.randint(12, 36),
            }
            e_priv = {
                "legal_review_outcome": rng.choice(["clean", "minor_issues"]),
                "liability_exposure_usd": rng.randint(10000, 400000),
                "has_auto_renewal": False,
                "termination_flexibility": rng.choice(["high", "medium"]),
                "ip_clause_favorable": True,
            }
            key_fields = ["legal_review_outcome", "liability_exposure_usd", "has_auto_renewal"]
        else:
            e_pub = {
                "contract_id": f"CTR-{instance_id}",
                "counterparty": rng.choice(["Acme Corp", "Global LLC", "TechStart Inc"]),
                "contract_value_usd": rng.randint(50000, 2000000),
                "duration_months": rng.randint(12, 60),
            }
            e_priv = {
                "legal_review_outcome": rng.choice(["major_issues", "unfavorable", "clean"]),
                "liability_exposure_usd": rng.randint(400000, 2000000),
                "has_auto_renewal": rng.choice([True, False]),
                "termination_flexibility": rng.choice(["low", "none"]),
                "ip_clause_favorable": rng.choice([True, False]),
            }
            key_fields = ["legal_review_outcome", "liability_exposure_usd", "has_auto_renewal", "termination_flexibility"]
        
        z = "Good" if is_good else "Bad"
        return TaskInstance(
            family=self.name,
            instance_id=instance_id,
            z=z,
            e_pub=e_pub,
            e_priv=e_priv,
            ground_truth_decision=self.ground_truth_rule(e_pub, e_priv),
            key_private_fields=key_fields,
        )


# ============ Registry ============
TASK_FAMILIES: Dict[str, TaskFamily] = {
    "project_approval": ProjectApproval(),
    "loan_approval": LoanApproval(),
    "experiment_replication": ExperimentReplication(),
    "security_triage": SecurityTriage(),
    "supplier_admission": SupplierAdmission(),
    "procurement_approval": ProcurementApproval(),
    "release_approval": ReleaseApproval(),
    "dataset_quality": DatasetQuality(),
    "medical_compliance": MedicalCompliance(),
    "contract_risk_screening": ContractRiskScreening(),
}


def generate_all_tasks(
    seed: int,
    families: int,
    episodes_per_family: int,
    family_offset: int = 0,
) -> List[TaskInstance]:
    """Generate all task instances for the experiment
    
    Args:
        seed: Random seed for reproducibility
        families: Number of families to generate (from start of TASK_FAMILIES)
        episodes_per_family: Number of episodes per family
        family_offset: Skip first N families (for parallel splitting)
    """
    rng = random.Random(seed)
    tasks = []
    
    all_family_names = list(TASK_FAMILIES.keys())
    family_names = all_family_names[family_offset : family_offset + families]
    
    for family_idx, family_name in enumerate(family_names):
        family = TASK_FAMILIES[family_name]
        for ep_idx in range(episodes_per_family):
            instance_id = f"{family_name}_{family_idx}_{ep_idx}"
            task = family.generate(rng, instance_id, ep_idx)
            tasks.append(task)
    
    rng.shuffle(tasks)
    return tasks


def evidence_to_text(e_pub: Dict, e_priv: Dict = None) -> str:
    """Convert evidence dict to readable text format"""
    lines = ["[PUBLIC EVIDENCE]"]
    for k, v in e_pub.items():
        lines.append(f"  {k}: {v}")
    
    if e_priv:
        lines.append("\n[PRIVATE EVIDENCE]")
        for k, v in e_priv.items():
            lines.append(f"  {k}: {v}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    tasks = generate_all_tasks(42, 10, 5)
    print(f"Generated {len(tasks)} tasks")
    for t in tasks[:3]:
        print(f"\n{t.family} | {t.instance_id} | z={t.z} | {t.ground_truth_decision}")
        print(evidence_to_text(t.e_pub, t.e_priv))
