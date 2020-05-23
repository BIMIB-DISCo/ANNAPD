def prediction(forget_dict, instances, corrects, epoch):
    """Code to replicate the Example Forgetting experiment
    """

    for instance, correct in zip(instances, corrects):
        instance_data = forget_dict.setdefault(
            instance,
            {'status': False, 'learning-events': [], 'forget-events': []})

        if correct and not instance_data['status']:
            # Learned
            instance_data['learning-events'].append(epoch)
            instance_data['status'] = True

        if not correct and instance_data['status']:
            # Forgotten
            instance_data['forget-events'].append(epoch)
            instance_data['status'] = False
