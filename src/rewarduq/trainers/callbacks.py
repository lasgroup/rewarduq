"""Custom callbacks for training reward models."""

from transformers import TrainerCallback


class EvaluateSaveCallback(TrainerCallback):
    """A TrainerCallback for evaluation and checkpointing at certain times during the training."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.needs_evaluate = False
        self.eval_on_epochs = sorted(getattr(args, "eval_on_epochs", None) or [])
        self.eval_on_end = getattr(args, "eval_on_end", False)

        self.needs_save = False
        self.save_on_epochs = sorted(getattr(args, "save_on_epochs", None) or [])
        self.save_on_end = getattr(args, "save_on_end", False)

    def on_step_begin(self, args, state, control, **kwargs):
        self.needs_evaluate = True
        self.needs_save = True
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        self.needs_evaluate = False
        return control

    def on_save(self, args, state, control, **kwargs):
        self.needs_save = False
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # Evaluate on certain epochs
        if len(self.eval_on_epochs) > 0 and state.epoch >= self.eval_on_epochs[0]:
            control.should_evaluate = self.needs_evaluate
            self.eval_on_epochs.pop(0)

        # Evaluate on end of training
        if self.eval_on_end and state.global_step >= state.max_steps:
            control.should_evaluate = self.needs_evaluate

        # Save on certain epochs
        if len(self.save_on_epochs) > 0 and state.epoch >= self.save_on_epochs[0]:
            control.should_save = self.needs_save
            self.save_on_epochs.pop(0)

        # Save on end of training
        if self.save_on_end and state.global_step >= state.max_steps:
            control.should_save = self.needs_save

        return control
